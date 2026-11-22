"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.

Data flow (B images, fixed ViT-B-32 shapes):
    [B,3,224,224] --patchify--> [B,49,3072] --project+CLS+pos--> [B,50,768]
        --12 x TransformerBlock--> [B,50,768] --CLS+post-Norm+proj--> [B,512]
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        # Step 1: split the feature axis D into (H, D/H); [B,N,D] -> [B,N,H,dh].
        # Heads are contiguous slices of the feature axis: feature f belongs to
        # head f // dh, so reshaping the last axis is already the correct cut.
        batch, tokens, width = x.shape
        split = x.reshape(batch, tokens, self.heads, self.head_dim)

        # Step 2: move the head axis next to the batch axis; [B,H,N,dh].
        # Each head keeps its own token sequence in the original order.
        return split.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        # Real k is 4D, so swap only its final two axes (dh <-> N) and keep the
        # batch/head groups untouched: [B,H,N,dh] x [B,H,dh,N] -> [B,H,N,N].
        scale = np.float32(np.sqrt(self.head_dim))
        scores = (q @ k.swapaxes(-1, -2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        # Each output row is a convex mixture of that head's value rows.
        context = coefficients @ v

        # Undo split_heads: [B,H,N,dh] -> [B,N,H,dh] -> [B,N,D]. The head axis
        # folds back into the feature axis in the same order used when splitting.
        # Unpack in the [B,H,N,dh] order that context actually uses: the second
        # axis is heads, the third is tokens. Swapping them would turn 50 tokens
        # into 12 and make the final reshape fail.
        batch, heads, tokens, head_dim = context.shape
        merged = context.transpose(0, 2, 1, 3).reshape(batch, tokens, heads * head_dim)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # The residual path keeps the *un-normalized* input x, not norm1(x).
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        # Weights are stored [out,in], so every linear layer multiplies by .T.
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)          # exact erf GELU, no approximation
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        batch = images.shape[0]
        patches = 224 // 32  # 7

        # Step 1: cut H and W into (grid, patch) pairs. The axis order becomes
        #         [B, C, 7, 32, 7, 32]; no data moves, only the indexing changes.
        grid = images.reshape(batch, 3, patches, 32, patches, 32)

        # Step 2: reorder to [B, 7, 7, C, 32, 32]. Row-major grid order (grid_h
        #         then grid_w) gives left-to-right, then top-to-bottom tokens;
        #         C before H before W gives the required within-patch flattening.
        grid = grid.transpose(0, 2, 4, 1, 3, 5)

        # Step 3: fold (7,7) into 49 token rows and (C,32,32) into 3072 features.
        return grid.reshape(batch, patches * patches, 3 * 32 * 32)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # conv1.weight was reshaped to [768,3072] = [out,in], hence .T.
        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        # axis=1 is the token axis: CLS becomes row 0, patch rows follow in order.
        tokens = np.concatenate([cls, projected], axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        # Token 0 is the CLS row: [B,50,768] -> [B,768].
        cls = tokens[:, 0, :]

        # A dedicated post-Norm, with parameters separate from ln_pre.
        normalized = self.post_norm(cls)

        # Already [in,out] = [768,512], so multiply without transposing.
        output = normalized @ self.projection

        return unit_vector(output)
