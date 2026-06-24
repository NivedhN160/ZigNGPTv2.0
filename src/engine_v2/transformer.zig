const std = @import("std");
const nn = @import("nn.zig");
const autograd = @import("autograd.zig");
const TensorNode = autograd.TensorNode;
const Allocator = std.mem.Allocator;

pub const SelfAttention = struct {
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    c_proj: nn.Linear,
    head_dim: usize,

    pub fn init(allocator: Allocator, embed_dim: usize) !SelfAttention {
        return SelfAttention{
            .q_proj = try nn.Linear.init(allocator, embed_dim, embed_dim),
            .k_proj = try nn.Linear.init(allocator, embed_dim, embed_dim),
            .v_proj = try nn.Linear.init(allocator, embed_dim, embed_dim),
            .c_proj = try nn.Linear.init(allocator, embed_dim, embed_dim),
            .head_dim = embed_dim,
        };
    }

    pub fn forward(self: *const SelfAttention, allocator: Allocator, x: *TensorNode) !*TensorNode {
        // x shape: [T, embed_dim]
        const q = try self.q_proj.forward(allocator, x); // [T, head_dim]
        const k = try self.k_proj.forward(allocator, x); // [T, head_dim]
        const v = try self.v_proj.forward(allocator, x); // [T, head_dim]

        const k_t = try TensorNode.transpose(allocator, k); // [head_dim, T]
        var scores = try TensorNode.matmul(allocator, q, k_t); // [T, T]

        // Scale by 1 / sqrt(head_dim)
        const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));
        var scale_data = try allocator.alloc(f32, 1);
        scale_data[0] = scale;
        const scale_node = try TensorNode.createData(allocator, .{1, 1}, scale_data);
        scores = try TensorNode.mul(allocator, scores, scale_node);

        // Apply causal mask
        scores = try TensorNode.causal_mask(allocator, scores);

        // Softmax
        const probs = try TensorNode.softmax(allocator, scores); // [T, T]

        // Multiply with values
        const out = try TensorNode.matmul(allocator, probs, v); // [T, head_dim]

        return try self.c_proj.forward(allocator, out);
    }

    pub fn parameters(self: *const SelfAttention, allocator: Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try self.q_proj.parameters(allocator, list);
        try self.k_proj.parameters(allocator, list);
        try self.v_proj.parameters(allocator, list);
        try self.c_proj.parameters(allocator, list);
    }
};

pub const FeedForward = struct {
    c_fc: nn.Linear,
    c_proj: nn.Linear,

    pub fn init(allocator: Allocator, embed_dim: usize) !FeedForward {
        return FeedForward{
            .c_fc = try nn.Linear.init(allocator, embed_dim, embed_dim * 4),
            .c_proj = try nn.Linear.init(allocator, embed_dim * 4, embed_dim),
        };
    }

    pub fn forward(self: *const FeedForward, allocator: Allocator, x: *TensorNode) !*TensorNode {
        const a = try self.c_fc.forward(allocator, x);
        const r = try TensorNode.relu(allocator, a);
        return try self.c_proj.forward(allocator, r);
    }

    pub fn parameters(self: *const FeedForward, allocator: Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try self.c_fc.parameters(allocator, list);
        try self.c_proj.parameters(allocator, list);
    }
};

pub const TransformerBlock = struct {
    ln_1: nn.LayerNorm,
    attn: SelfAttention,
    ln_2: nn.LayerNorm,
    mlp: FeedForward,

    pub fn init(allocator: Allocator, embed_dim: usize) !TransformerBlock {
        return TransformerBlock{
            .ln_1 = try nn.LayerNorm.init(allocator, embed_dim),
            .attn = try SelfAttention.init(allocator, embed_dim),
            .ln_2 = try nn.LayerNorm.init(allocator, embed_dim),
            .mlp = try FeedForward.init(allocator, embed_dim),
        };
    }

    pub fn forward(self: *const TransformerBlock, allocator: Allocator, x: *TensorNode) !*TensorNode {
        const norm1 = try self.ln_1.forward(allocator, x);
        const a = try self.attn.forward(allocator, norm1);
        const out1 = try TensorNode.add(allocator, x, a);

        const norm2 = try self.ln_2.forward(allocator, out1);
        const m = try self.mlp.forward(allocator, norm2);
        return try TensorNode.add(allocator, out1, m);
    }

    pub fn parameters(self: *const TransformerBlock, allocator: Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try self.ln_1.parameters(allocator, list);
        try self.attn.parameters(allocator, list);
        try self.ln_2.parameters(allocator, list);
        try self.mlp.parameters(allocator, list);
    }
};

pub const Transformer = struct {
    wte: nn.Embedding,
    wpe: nn.Embedding,
    blocks: std.ArrayList(TransformerBlock),
    ln_f: nn.LayerNorm,
    lm_head: nn.Linear,

    pub fn init(allocator: Allocator, vocab_size: usize, embed_dim: usize, num_layers: usize, max_seq_len: usize) !Transformer {
        var blocks = std.ArrayList(TransformerBlock).empty;
        for (0..num_layers) |_| {
            try blocks.append(allocator, try TransformerBlock.init(allocator, embed_dim));
        }

        return Transformer{
            .wte = try nn.Embedding.init(allocator, vocab_size, embed_dim),
            .wpe = try nn.Embedding.init(allocator, max_seq_len, embed_dim),
            .blocks = blocks,
            .ln_f = try nn.LayerNorm.init(allocator, embed_dim),
            .lm_head = try nn.Linear.init(allocator, embed_dim, vocab_size),
        };
    }

    pub fn deinit(self: *Transformer, allocator: Allocator) void {
        self.blocks.deinit(allocator);
    }

    pub fn forward(self: *const Transformer, allocator: Allocator, indices: []const usize) !*TensorNode {
        const seq_len = indices.len;
        const tok_emb = try self.wte.forward(allocator, indices);

        var pos_indices = try allocator.alloc(usize, seq_len);
        for (0..seq_len) |i| pos_indices[i] = i;
        const pos_emb = try self.wpe.forward(allocator, pos_indices);

        var x = try TensorNode.add(allocator, tok_emb, pos_emb);

        for (self.blocks.items) |block| {
            x = try block.forward(allocator, x);
        }

        x = try self.ln_f.forward(allocator, x);
        return try self.lm_head.forward(allocator, x);
    }

    pub fn parameters(self: *const Transformer, allocator: Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try self.wte.parameters(allocator, list);
        try self.wpe.parameters(allocator, list);
        for (self.blocks.items) |block| {
            try block.parameters(allocator, list);
        }
        try self.ln_f.parameters(allocator, list);
        try self.lm_head.parameters(allocator, list);
    }
};

test "Transformer build check" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var model = try Transformer.init(aa, 100, 32, 2, 64);
    
    var indices = [_]usize{ 1, 5, 20, 99 };
    const out = try model.forward(aa, &indices);
    
    try std.testing.expectEqual(@as(usize, 4), out.data.shape[0]);
    try std.testing.expectEqual(@as(usize, 100), out.data.shape[1]);
}
