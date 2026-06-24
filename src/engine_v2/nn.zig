const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const autograd = @import("autograd.zig");
const TensorNode = autograd.TensorNode;

pub const Linear = struct {
    weight: *TensorNode,
    bias: ?*TensorNode,

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Linear {
        var rng = std.Random.DefaultPrng.init(0);
        const random = rng.random();
        const std_dev = std.math.sqrt(2.0 / @as(f32, @floatFromInt(in_features)));

        const w_data = try allocator.alloc(f32, in_features * out_features);
        defer allocator.free(w_data);
        for (0..in_features * out_features) |i| {
            w_data[i] = @as(f32, @floatCast(random.floatNorm(f64))) * std_dev;
        }

        const b_data = try allocator.alloc(f32, out_features);
        defer allocator.free(b_data);
        @memset(b_data, 0.0);

        return Linear{
            .weight = try TensorNode.createData(allocator, .{in_features, out_features}, w_data),
            .bias = try TensorNode.createData(allocator, .{1, out_features}, b_data),
        };
    }

    pub fn forward(self: *const Linear, allocator: std.mem.Allocator, x: *TensorNode) !*TensorNode {
        const xw = try TensorNode.matmul(allocator, x, self.weight);
        if (self.bias) |b| {
            return try TensorNode.add(allocator, xw, b);
        }
        return xw;
    }

    pub fn save(self: *const Linear, file: std.fs.File) !void {
        try self.weight.save(file);
        if (self.bias) |b| {
            try b.save(file);
        }
    }

    pub fn load(self: *Linear, file: std.fs.File) !void {
        try self.weight.load(file);
        if (self.bias) |b| {
            try b.load(file);
        }
    }

    pub fn parameters(self: *const Linear, allocator: std.mem.Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try list.append(allocator, self.weight);
        if (self.bias) |b| try list.append(allocator, b);
    }
};

pub const Embedding = struct {
    weight: *TensorNode,

    pub fn init(allocator: std.mem.Allocator, num_embeddings: usize, embedding_dim: usize) !Embedding {
        var rng = std.Random.DefaultPrng.init(0);
        const random = rng.random();
        
        const w_data = try allocator.alloc(f32, num_embeddings * embedding_dim);
        defer allocator.free(w_data);
        for (0..num_embeddings * embedding_dim) |i| {
            w_data[i] = @as(f32, @floatCast(random.floatNorm(f64)));
        }

        return Embedding{
            .weight = try TensorNode.createData(allocator, .{num_embeddings, embedding_dim}, w_data),
        };
    }

    pub fn forward(self: *const Embedding, allocator: std.mem.Allocator, indices: []const usize) !*TensorNode {
        return try TensorNode.embedding(allocator, self.weight, indices);
    }

    pub fn save(self: *const Embedding, file: std.fs.File) !void {
        try self.weight.save(file);
    }

    pub fn load(self: *Embedding, file: std.fs.File) !void {
        try self.weight.load(file);
    }

    pub fn parameters(self: *const Embedding, allocator: std.mem.Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try list.append(allocator, self.weight);
    }
};

pub const LayerNorm = struct {
    gamma: *TensorNode,
    beta: *TensorNode,

    pub fn init(allocator: std.mem.Allocator, normalized_shape: usize) !LayerNorm {
        const gamma_data = try allocator.alloc(f32, normalized_shape);
        defer allocator.free(gamma_data);
        @memset(gamma_data, 1.0);

        const beta_data = try allocator.alloc(f32, normalized_shape);
        defer allocator.free(beta_data);
        @memset(beta_data, 0.0);

        return LayerNorm{
            .gamma = try TensorNode.createData(allocator, .{1, normalized_shape}, gamma_data),
            .beta = try TensorNode.createData(allocator, .{1, normalized_shape}, beta_data),
        };
    }

    pub fn forward(self: *const LayerNorm, allocator: std.mem.Allocator, x: *TensorNode) !*TensorNode {
        const norm = try TensorNode.normalize(allocator, x);
        const norm_gamma = try TensorNode.mul(allocator, norm, self.gamma);
        return try TensorNode.add(allocator, norm_gamma, self.beta);
    }

    pub fn save(self: *const LayerNorm, file: std.fs.File) !void {
        try self.gamma.save(file);
        try self.beta.save(file);
    }

    pub fn load(self: *LayerNorm, file: std.fs.File) !void {
        try self.gamma.load(file);
        try self.beta.load(file);
    }

    pub fn parameters(self: *const LayerNorm, allocator: std.mem.Allocator, list: *std.ArrayList(*TensorNode)) !void {
        try list.append(allocator, self.gamma);
        try list.append(allocator, self.beta);
    }
};

pub const SGD = struct {
    params: std.ArrayList(*TensorNode),
    allocator: std.mem.Allocator,
    lr: f32,

    pub fn init(allocator: std.mem.Allocator, lr: f32) SGD {
        return SGD{
            .params = std.ArrayList(*TensorNode).empty,
            .allocator = allocator,
            .lr = lr,
        };
    }

    pub fn deinit(self: *SGD) void {
        self.params.deinit(self.allocator);
    }

    pub fn step(self: *SGD) void {
        for (self.params.items) |p| {
            for (0..p.data.data.len) |i| {
                p.data.data[i] -= self.lr * p.grad.data[i];
            }
        }
    }

    pub fn zeroGrad(self: *SGD) void {
        for (self.params.items) |p| {
            @memset(p.grad.data, 0.0);
        }
    }
};

const testing = std.testing;

test "XOR Overfit" {
    var arena_main = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_main.deinit();
    const alloc = arena_main.allocator();

    const linear1 = try Linear.init(alloc, 2, 8);
    const linear2 = try Linear.init(alloc, 8, 1);

    var sgd = SGD.init(alloc, 0.1);
    defer sgd.deinit();
    try linear1.parameters(alloc, &sgd.params);
    try linear2.parameters(alloc, &sgd.params);

    // XOR data
    const x_data = [_]f32{
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const y_data = [_]f32{
        0.0,
        1.0,
        1.0,
        0.0,
    };

    var final_loss: f32 = std.math.inf(f32);

    for (0..5000) |epoch| {
        _ = epoch; // Use if we want to print
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const a_alloc = arena.allocator();
        
        const x_node = try TensorNode.createData(a_alloc, .{4, 2}, &x_data);
        const y_node = try TensorNode.createData(a_alloc, .{4, 1}, &y_data);

        const h1 = try linear1.forward(a_alloc, x_node);
        const h1_relu = try TensorNode.relu(a_alloc, h1);
        const y_pred = try linear2.forward(a_alloc, h1_relu);

        const loss = try TensorNode.mse_loss(a_alloc, y_pred, y_node);
        
        final_loss = loss.data.data[0];

        sgd.zeroGrad();
        try loss.backward(a_alloc);
        sgd.step();
    }

    std.debug.print("\nFinal loss: {}\n", .{final_loss});
    try testing.expect(final_loss < 0.1);
}
