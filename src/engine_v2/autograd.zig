const std = @import("std");

pub const OpType = enum {
    none,
    add,
    mul,
    relu,
};

pub const Value = struct {
    data: f32,
    grad: f32,
    op: OpType,
    lhs: ?*Value,
    rhs: ?*Value,

    pub fn create(allocator: std.mem.Allocator, data: f32) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .data = data,
            .grad = 0.0,
            .op = .none,
            .lhs = null,
            .rhs = null,
        };
        return val;
    }

    pub fn add(allocator: std.mem.Allocator, a: *Value, b: *Value) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .data = a.data + b.data,
            .grad = 0.0,
            .op = .add,
            .lhs = a,
            .rhs = b,
        };
        return val;
    }

    pub fn mul(allocator: std.mem.Allocator, a: *Value, b: *Value) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .data = a.data * b.data,
            .grad = 0.0,
            .op = .mul,
            .lhs = a,
            .rhs = b,
        };
        return val;
    }

    pub fn relu(allocator: std.mem.Allocator, a: *Value) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .data = @max(0.0, a.data),
            .grad = 0.0,
            .op = .relu,
            .lhs = a,
            .rhs = null,
        };
        return val;
    }

    fn backwardStep(self: *Value) void {
        switch (self.op) {
            .none => {},
            .add => {
                if (self.lhs) |l| l.grad += self.grad;
                if (self.rhs) |r| r.grad += self.grad;
            },
            .mul => {
                if (self.lhs) |l| l.grad += self.rhs.?.data * self.grad;
                if (self.rhs) |r| r.grad += self.lhs.?.data * self.grad;
            },
            .relu => {
                if (self.lhs) |l| {
                    if (self.data > 0.0) {
                        l.grad += self.grad;
                    }
                }
            },
        }
    }

    fn buildTopo(allocator: std.mem.Allocator, node: *Value, topo: *std.ArrayList(*Value), visited: *std.AutoHashMap(*Value, void)) !void {
        if (!visited.contains(node)) {
            try visited.put(node, {});
            if (node.lhs) |l| try buildTopo(allocator, l, topo, visited);
            if (node.rhs) |r| try buildTopo(allocator, r, topo, visited);
            try topo.append(allocator, node);
        }
    }

    pub fn backward(self: *Value, allocator: std.mem.Allocator) !void {
        var topo = std.ArrayList(*Value).empty;
        defer topo.deinit(allocator);

        var visited = std.AutoHashMap(*Value, void).init(allocator);
        defer visited.deinit();

        try buildTopo(allocator, self, &topo, &visited);

        self.grad = 1.0;
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            topo.items[i].backwardStep();
        }
    }
};

const Tensor = @import("tensor.zig").Tensor;

pub const TensorOpType = enum {
    none,
    add,
    matmul,
    softmax,
    relu,
    mse_loss,
    embedding,
    normalize,
    mul,
    causal_mask,
    transpose,
};

pub const TensorNode = struct {
    data: Tensor,
    grad: Tensor,
    op: TensorOpType,
    lhs: ?*TensorNode = null,
    rhs: ?*TensorNode = null,
    indices: ?[]const usize = null,

    pub fn create(allocator: std.mem.Allocator, shape: [2]usize) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try Tensor.init(allocator, shape),
            .grad = try Tensor.init(allocator, shape),
            .op = .none,
            .lhs = null,
            .rhs = null,
        };
        return node;
    }

    pub fn createData(allocator: std.mem.Allocator, shape: [2]usize, data: []const f32) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try Tensor.initData(allocator, shape, data),
            .grad = try Tensor.init(allocator, shape),
            .op = .none,
            .lhs = null,
            .rhs = null,
        };
        return node;
    }

    pub fn add(allocator: std.mem.Allocator, a: *TensorNode, b: *TensorNode) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try a.data.add(&b.data),
            .grad = try Tensor.init(allocator, a.data.shape),
            .op = .add,
            .lhs = a,
            .rhs = b,
        };
        return node;
    }

    pub fn mul(allocator: std.mem.Allocator, lhs: *TensorNode, rhs: *TensorNode) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try lhs.data.mul(&rhs.data),
            .grad = try Tensor.init(allocator, lhs.data.shape),
            .op = .mul,
            .lhs = lhs,
            .rhs = rhs,
        };
        return node;
    }

    pub fn causal_mask(allocator: std.mem.Allocator, lhs: *TensorNode) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try lhs.data.causal_mask(allocator),
            .grad = try Tensor.init(allocator, lhs.data.shape),
            .op = .causal_mask,
            .lhs = lhs,
            .rhs = null,
        };
        return node;
    }

    pub fn transpose(allocator: std.mem.Allocator, lhs: *TensorNode) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try lhs.data.transpose(),
            .grad = try Tensor.init(allocator, .{lhs.data.shape[1], lhs.data.shape[0]}),
            .op = .transpose,
            .lhs = lhs,
            .rhs = null,
        };
        return node;
    }

    pub fn matmul(allocator: std.mem.Allocator, a: *TensorNode, b: *TensorNode) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = try a.data.matmul(&b.data),
            .grad = try Tensor.init(allocator, [2]usize{a.data.shape[0], b.data.shape[1]}),
            .op = .matmul,
            .lhs = a,
            .rhs = b,
        };
        return node;
    }

    pub fn softmax(allocator: std.mem.Allocator, a: *TensorNode) !*TensorNode {
        var out_data = try Tensor.init(allocator, a.data.shape);
        for (0..a.data.shape[0]) |i| {
            var max_val: f32 = -std.math.inf(f32);
            for (0..a.data.shape[1]) |j| {
                max_val = @max(max_val, a.data.get(i, j));
            }
            var sum_exp: f32 = 0.0;
            for (0..a.data.shape[1]) |j| {
                const e = std.math.exp(a.data.get(i, j) - max_val);
                out_data.set(i, j, e);
                sum_exp += e;
            }
            for (0..a.data.shape[1]) |j| {
                out_data.set(i, j, out_data.get(i, j) / sum_exp);
            }
        }

        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = out_data,
            .grad = try Tensor.init(allocator, a.data.shape),
            .op = .softmax,
            .lhs = a,
            .rhs = null,
        };
        return node;
    }

    pub fn relu(allocator: std.mem.Allocator, a: *TensorNode) !*TensorNode {
        var out_data = try Tensor.init(allocator, a.data.shape);
        for (0..a.data.data.len) |i| {
            out_data.data[i] = @max(0.0, a.data.data[i]);
        }
        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = out_data,
            .grad = try Tensor.init(allocator, a.data.shape),
            .op = .relu,
            .lhs = a,
            .rhs = null,
        };
        return node;
    }

    pub fn mse_loss(allocator: std.mem.Allocator, pred: *TensorNode, target: *TensorNode) !*TensorNode {
        var loss_val: f32 = 0.0;
        const n = @as(f32, @floatFromInt(pred.data.data.len));
        for (0..pred.data.data.len) |i| {
            const diff = pred.data.data[i] - target.data.data[i];
            loss_val += diff * diff;
        }
        loss_val /= n;

        var out_data = try Tensor.init(allocator, .{1, 1});
        out_data.data[0] = loss_val;

        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = out_data,
            .grad = try Tensor.init(allocator, .{1, 1}),
            .op = .mse_loss,
            .lhs = pred,
            .rhs = target,
        };
        return node;
    }

    pub fn embedding(allocator: std.mem.Allocator, weight: *TensorNode, indices: []const usize) !*TensorNode {
        const batch_size = indices.len;
        const embed_dim = weight.data.shape[1];
        var out_data = try Tensor.init(allocator, .{batch_size, embed_dim});
        
        for (0..batch_size) |i| {
            const idx = indices[i];
            for (0..embed_dim) |j| {
                out_data.set(i, j, weight.data.get(idx, j));
            }
        }

        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = out_data,
            .grad = try Tensor.init(allocator, .{batch_size, embed_dim}),
            .op = .embedding,
            .lhs = weight,
            .rhs = null,
            .indices = try allocator.dupe(usize, indices),
        };
        return node;
    }

    pub fn normalize(allocator: std.mem.Allocator, x: *TensorNode) !*TensorNode {
        var out_data = try Tensor.init(allocator, x.data.shape);
        const eps = 1e-5;
        const cols = @as(f32, @floatFromInt(x.data.shape[1]));

        for (0..x.data.shape[0]) |i| {
            var sum: f32 = 0;
            for (0..x.data.shape[1]) |j| sum += x.data.get(i, j);
            const mean = sum / cols;

            var var_sum: f32 = 0;
            for (0..x.data.shape[1]) |j| {
                const diff = x.data.get(i, j) - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / cols;
            const std_dev = std.math.sqrt(variance + eps);

            for (0..x.data.shape[1]) |j| {
                const norm_val = (x.data.get(i, j) - mean) / std_dev;
                out_data.set(i, j, norm_val);
            }
        }

        const node = try allocator.create(TensorNode);
        node.* = .{
            .data = out_data,
            .grad = try Tensor.init(allocator, x.data.shape),
            .op = .normalize,
            .lhs = x,
            .rhs = null,
        };
        return node;
    }

    fn backwardStep(self: *TensorNode) !void {
        switch (self.op) {
            .none => {},
            .add => {
                if (self.lhs) |l| {
                    if (l.data.shape[0] == 1 and self.grad.shape[0] > 1) {
                        for (0..self.grad.shape[0]) |i| {
                            for (0..self.grad.shape[1]) |j| {
                                l.grad.set(0, j, l.grad.get(0, j) + self.grad.get(i, j));
                            }
                        }
                    } else if (l.data.shape[1] == 1 and self.grad.shape[1] > 1) {
                        for (0..self.grad.shape[0]) |i| {
                            for (0..self.grad.shape[1]) |j| {
                                l.grad.set(i, 0, l.grad.get(i, 0) + self.grad.get(i, j));
                            }
                        }
                    } else {
                        for (0..l.grad.data.len) |i| l.grad.data[i] += self.grad.data[i];
                    }
                }
                if (self.rhs) |r| {
                    if (r.data.shape[0] == 1 and self.grad.shape[0] > 1) {
                        for (0..self.grad.shape[0]) |i| {
                            for (0..self.grad.shape[1]) |j| {
                                r.grad.set(0, j, r.grad.get(0, j) + self.grad.get(i, j));
                            }
                        }
                    } else if (r.data.shape[1] == 1 and self.grad.shape[1] > 1) {
                        for (0..self.grad.shape[0]) |i| {
                            for (0..self.grad.shape[1]) |j| {
                                r.grad.set(i, 0, r.grad.get(i, 0) + self.grad.get(i, j));
                            }
                        }
                    } else {
                        for (0..r.grad.data.len) |i| r.grad.data[i] += self.grad.data[i];
                    }
                }
            },
            .matmul => {
                if (self.lhs) |l| {
                    var b_t = try self.rhs.?.data.transpose();
                    defer b_t.deinit();
                    var dl_da = try self.grad.matmul(&b_t);
                    defer dl_da.deinit();
                    for (0..l.grad.data.len) |i| l.grad.data[i] += dl_da.data[i];
                }
                if (self.rhs) |r| {
                    var a_t = try self.lhs.?.data.transpose();
                    defer a_t.deinit();
                    var dl_db = try a_t.matmul(&self.grad);
                    defer dl_db.deinit();
                    for (0..r.grad.data.len) |i| r.grad.data[i] += dl_db.data[i];
                }
            },
            .softmax => {
                if (self.lhs) |l| {
                    for (0..self.data.shape[0]) |i| {
                        var sum_grad_out: f32 = 0.0;
                        for (0..self.data.shape[1]) |j| {
                            sum_grad_out += self.data.get(i, j) * self.grad.get(i, j);
                        }
                        for (0..self.data.shape[1]) |j| {
                            const val = self.data.get(i, j);
                            const g = self.grad.get(i, j);
                            const dl_din = val * (g - sum_grad_out);
                            l.grad.set(i, j, l.grad.get(i, j) + dl_din);
                        }
                    }
                }
            },
            .relu => {
                if (self.lhs) |l| {
                    for (0..self.data.data.len) |i| {
                        if (self.data.data[i] > 0.0) {
                            l.grad.data[i] += self.grad.data[i];
                        }
                    }
                }
            },
            .mse_loss => {
                if (self.lhs) |l| {
                    const n = @as(f32, @floatFromInt(l.data.data.len));
                    for (0..l.data.data.len) |i| {
                        const diff = l.data.data[i] - self.rhs.?.data.data[i];
                        l.grad.data[i] += self.grad.data[0] * 2.0 * diff / n;
                    }
                }
            },
            .mul => {
                if (self.lhs) |l| {
                    for (0..self.grad.shape[0]) |i| {
                        for (0..self.grad.shape[1]) |j| {
                            const r_i = if (self.rhs.?.data.shape[0] == 1) 0 else i;
                            const r_j = if (self.rhs.?.data.shape[1] == 1) 0 else j;
                            const r_val = self.rhs.?.data.get(r_i, r_j);
                            
                            const l_i = if (l.data.shape[0] == 1) 0 else i;
                            const l_j = if (l.data.shape[1] == 1) 0 else j;
                            
                            l.grad.set(l_i, l_j, l.grad.get(l_i, l_j) + self.grad.get(i, j) * r_val);
                        }
                    }
                }
                if (self.rhs) |r| {
                    for (0..self.grad.shape[0]) |i| {
                        for (0..self.grad.shape[1]) |j| {
                            const l_i = if (self.lhs.?.data.shape[0] == 1) 0 else i;
                            const l_j = if (self.lhs.?.data.shape[1] == 1) 0 else j;
                            const l_val = self.lhs.?.data.get(l_i, l_j);
                            
                            const r_i = if (r.data.shape[0] == 1) 0 else i;
                            const r_j = if (r.data.shape[1] == 1) 0 else j;
                            
                            r.grad.set(r_i, r_j, r.grad.get(r_i, r_j) + self.grad.get(i, j) * l_val);
                        }
                    }
                }
            },
            .embedding => {
                if (self.lhs) |l| {
                    if (self.indices) |inds| {
                        for (0..inds.len) |i| {
                            const idx = inds[i];
                            for (0..self.grad.shape[1]) |j| {
                                l.grad.set(idx, j, l.grad.get(idx, j) + self.grad.get(i, j));
                            }
                        }
                    }
                }
            },
            .normalize => {
                if (self.lhs) |l| {
                    const eps = 1e-5;
                    const cols = @as(f32, @floatFromInt(l.data.shape[1]));

                    for (0..l.data.shape[0]) |i| {
                        var sum_x: f32 = 0;
                        for (0..l.data.shape[1]) |j| sum_x += l.data.get(i, j);
                        const mean = sum_x / cols;

                        var var_sum: f32 = 0;
                        for (0..l.data.shape[1]) |j| {
                            const diff = l.data.get(i, j) - mean;
                            var_sum += diff * diff;
                        }
                        const variance = var_sum / cols;
                        const std_dev = std.math.sqrt(variance + eps);

                        var sum_dx: f32 = 0;
                        var sum_dx_x_hat: f32 = 0;

                        for (0..l.data.shape[1]) |j| {
                            const dx = self.grad.get(i, j);
                            const x_hat = self.data.get(i, j);
                            sum_dx += dx;
                            sum_dx_x_hat += dx * x_hat;
                        }

                        for (0..l.data.shape[1]) |j| {
                            const x_hat = self.data.get(i, j);
                            const dx = self.grad.get(i, j);
                            
                            const grad_in = (1.0 / std_dev) * (dx - (sum_dx / cols) - x_hat * (sum_dx_x_hat / cols));
                            l.grad.set(i, j, l.grad.get(i, j) + grad_in);
                        }
                    }
                }
            },
            .causal_mask => {
                if (self.lhs) |l| {
                    const n = self.data.shape[0];
                    for (0..n) |i| {
                        for (0..n) |j| {
                            if (j <= i) {
                                l.grad.set(i, j, l.grad.get(i, j) + self.grad.get(i, j));
                            }
                        }
                    }
                }
            },
            .transpose => {
                if (self.lhs) |l| {
                    for (0..self.grad.shape[0]) |i| {
                        for (0..self.grad.shape[1]) |j| {
                            l.grad.set(j, i, l.grad.get(j, i) + self.grad.get(i, j));
                        }
                    }
                }
            },
        }
    }

    fn buildTopo(allocator: std.mem.Allocator, node: *TensorNode, topo: *std.ArrayList(*TensorNode), visited: *std.AutoHashMap(*TensorNode, void)) !void {
        if (!visited.contains(node)) {
            try visited.put(node, {});
            if (node.lhs) |l| try buildTopo(allocator, l, topo, visited);
            if (node.rhs) |r| try buildTopo(allocator, r, topo, visited);
            try topo.append(allocator, node);
        }
    }

    pub fn backward(self: *TensorNode, allocator: std.mem.Allocator) !void {
        var topo = std.ArrayList(*TensorNode).empty;
        defer topo.deinit(allocator);

        var visited = std.AutoHashMap(*TensorNode, void).init(allocator);
        defer visited.deinit();

        try buildTopo(allocator, self, &topo, &visited);

        @memset(self.grad.data, 1.0);
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            try topo.items[i].backwardStep();
        }
    }

    pub fn save(self: *TensorNode, file: std.fs.File) !void {
        const bytes = std.mem.sliceAsBytes(self.data.data);
        try file.writeAll(bytes);
    }

    pub fn load(self: *TensorNode, file: std.fs.File) !void {
        const bytes = std.mem.sliceAsBytes(self.data.data);
        _ = try file.readAll(bytes);
    }
};

const testing = std.testing;

fn expectApproxEq(actual: f32, expected: f32, tol: f32) !void {
    try testing.expectApproxEqAbs(expected, actual, tol);
}

test "Autograd add" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try Value.create(alloc, 2.0);
    const b = try Value.create(alloc, -3.0);
    const c = try Value.add(alloc, a, b);
    try c.backward(alloc);

    try expectApproxEq(a.grad, 1.0, 1e-5);
    try expectApproxEq(b.grad, 1.0, 1e-5);
}

test "Autograd mul" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try Value.create(alloc, 2.0);
    const b = try Value.create(alloc, -3.0);
    const c = try Value.mul(alloc, a, b);
    try c.backward(alloc);

    try expectApproxEq(a.grad, -3.0, 1e-5);
    try expectApproxEq(b.grad, 2.0, 1e-5);
}

test "Autograd relu" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try Value.create(alloc, -2.0);
    const c = try Value.relu(alloc, a);
    try c.backward(alloc);

    try expectApproxEq(a.grad, 0.0, 1e-5);

    const a2 = try Value.create(alloc, 2.0);
    const c2 = try Value.relu(alloc, a2);
    try c2.backward(alloc);

    try expectApproxEq(a2.grad, 1.0, 1e-5);
}

test "Autograd complex graph with numerical checking" {
    // f(a, b) = relu(a * b + b)
    const buildGraph = struct {
        fn func(alloc: std.mem.Allocator, a: *Value, b: *Value) !*Value {
            const mul_res = try Value.mul(alloc, a, b);
            const add_res = try Value.add(alloc, mul_res, b);
            const relu_res = try Value.relu(alloc, add_res);
            return relu_res;
        }
    }.func;

    // 1. Analytical
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    
    const a = try Value.create(alloc, 2.0);
    const b = try Value.create(alloc, 3.0);
    const out = try buildGraph(alloc, a, b);
    try out.backward(alloc);

    const grad_a = a.grad;
    const grad_b = b.grad;

    // 2. Numerical
    const eps: f32 = 1e-3;
    var arena2 = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena2.deinit();
    const alloc2 = arena2.allocator();
    const a_plus = try Value.create(alloc2, 2.0 + eps);
    const b_orig1 = try Value.create(alloc2, 3.0);
    const out_a_plus = try buildGraph(alloc2, a_plus, b_orig1);

    var arena3 = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena3.deinit();
    const alloc3 = arena3.allocator();
    const a_minus = try Value.create(alloc3, 2.0 - eps);
    const b_orig2 = try Value.create(alloc3, 3.0);
    const out_a_minus = try buildGraph(alloc3, a_minus, b_orig2);

    const num_grad_a = (out_a_plus.data - out_a_minus.data) / (2.0 * eps);

    var arena4 = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena4.deinit();
    const alloc4 = arena4.allocator();
    const a_orig3 = try Value.create(alloc4, 2.0);
    const b_plus = try Value.create(alloc4, 3.0 + eps);
    const out_b_plus = try buildGraph(alloc4, a_orig3, b_plus);

    var arena5 = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena5.deinit();
    const alloc5 = arena5.allocator();
    const a_orig4 = try Value.create(alloc5, 2.0);
    const b_minus = try Value.create(alloc5, 3.0 - eps);
    const out_b_minus = try buildGraph(alloc5, a_orig4, b_minus);

    const num_grad_b = (out_b_plus.data - out_b_minus.data) / (2.0 * eps);

    try expectApproxEq(grad_a, num_grad_a, 1e-2);
    try expectApproxEq(grad_b, num_grad_b, 1e-2);
}

fn expectTensorApproxEq(actual: Tensor, expected: Tensor, tol: f32) !void {
    try testing.expectEqual(actual.shape[0], expected.shape[0]);
    try testing.expectEqual(actual.shape[1], expected.shape[1]);
    for (0..actual.data.len) |i| {
        try testing.expectApproxEqAbs(expected.data[i], actual.data[i], tol);
    }
}

test "TensorNode add" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var a_data = [_]f32{1.0, 2.0, 3.0, 4.0};
    const a = try TensorNode.createData(alloc, .{2, 2}, &a_data);

    var b_data = [_]f32{5.0, 6.0, 7.0, 8.0};
    const b = try TensorNode.createData(alloc, .{2, 2}, &b_data);

    const c = try TensorNode.add(alloc, a, b);
    try c.backward(alloc);

    for (0..4) |i| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), a.grad.data[i], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 1.0), b.grad.data[i], 1e-5);
    }
}

test "TensorNode matmul" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var a_data = [_]f32{1.0, 2.0, 3.0, 4.0};
    const a = try TensorNode.createData(alloc, .{2, 2}, &a_data);

    var b_data = [_]f32{5.0, 6.0, 7.0, 8.0};
    const b = try TensorNode.createData(alloc, .{2, 2}, &b_data);

    const c = try TensorNode.matmul(alloc, a, b);
    try c.backward(alloc);

    // c = a * b
    // dc/da = 1 * b^T. 
    // Wait, the "ones" grad represents sum over elements. 
    // c_ij = sum_k a_ik * b_kj. Sum(c_ij) = sum_i sum_j sum_k a_ik b_kj
    // d(Sum)/d(a_pq) = sum_j b_qj.
    // Let's manually verify:
    // b = [[5, 6], [7, 8]]
    // a.grad for row p, col q is sum_j b_qj = 5+6=11 for q=0, 7+8=15 for q=1.
    // So a.grad should be [[11, 15], [11, 15]]
    try testing.expectApproxEqAbs(@as(f32, 11.0), a.grad.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 15.0), a.grad.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 11.0), a.grad.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 15.0), a.grad.data[3], 1e-5);

    // d(Sum)/d(b_pq) = sum_i a_ip
    // a = [[1, 2], [3, 4]]
    // b.grad for row p, col q is sum_i a_ip = 1+3=4 for p=0, 2+4=6 for p=1.
    // So b.grad should be [[4, 4], [6, 6]]
    try testing.expectApproxEqAbs(@as(f32, 4.0), b.grad.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.0), b.grad.data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.0), b.grad.data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.0), b.grad.data[3], 1e-5);
}

test "TensorNode softmax numerical check" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var a_data = [_]f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const a = try TensorNode.createData(alloc, .{2, 3}, &a_data);

    // We do a sum over softmax outputs as the loss:
    const buildGraph = struct {
        fn func(a_alloc: std.mem.Allocator, input: *TensorNode) !f32 {
            const sm = try TensorNode.softmax(a_alloc, input);
            var sum: f32 = 0;
            for (sm.data.data) |v| sum += v;
            return sum;
        }
    }.func;

    const sm = try TensorNode.softmax(alloc, a);
    try sm.backward(alloc);

    // numerical
    const eps = 1e-3;
    var num_grad = try Tensor.init(alloc, .{2, 3});
    for (0..6) |i| {
        var a_plus_data = a_data;
        a_plus_data[i] += eps;
        var a_minus_data = a_data;
        a_minus_data[i] -= eps;

        const a_plus = try TensorNode.createData(alloc, .{2, 3}, &a_plus_data);
        const a_minus = try TensorNode.createData(alloc, .{2, 3}, &a_minus_data);

        const out_plus = try buildGraph(alloc, a_plus);
        const out_minus = try buildGraph(alloc, a_minus);

        num_grad.data[i] = (out_plus - out_minus) / (2.0 * eps);
    }

    for (0..6) |i| {
        try testing.expectApproxEqAbs(num_grad.data[i], a.grad.data[i], 1e-3);
    }
}

