const std = @import("std");

pub const Tensor = struct {
    data: []f32,
    shape: [2]usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, shape: [2]usize) !Tensor {
        const size = shape[0] * shape[1];
        const data = try allocator.alloc(f32, size);
        @memset(data, 0.0);
        return Tensor{
            .data = data,
            .shape = shape,
            .allocator = allocator,
        };
    }

    pub fn initData(allocator: std.mem.Allocator, shape: [2]usize, data: []const f32) !Tensor {
        std.debug.assert(data.len == shape[0] * shape[1]);
        const allocated_data = try allocator.dupe(f32, data);
        return Tensor{
            .data = allocated_data,
            .shape = shape,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: *const Tensor, row: usize, col: usize) f32 {
        std.debug.assert(row < self.shape[0] and col < self.shape[1]);
        return self.data[row * self.shape[1] + col];
    }

    pub fn set(self: *Tensor, row: usize, col: usize, val: f32) void {
        std.debug.assert(row < self.shape[0] and col < self.shape[1]);
        self.data[row * self.shape[1] + col] = val;
    }

    pub fn add(self: *const Tensor, other: *const Tensor) !Tensor {
        if (self.shape[0] == other.shape[0] and self.shape[1] == other.shape[1]) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (self.data, 0..) |v, i| {
                out.data[i] = v + other.data[i];
            }
            return out;
        } else if (self.shape[1] == other.shape[1] and other.shape[0] == 1) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (0..self.shape[0]) |i| {
                for (0..self.shape[1]) |j| {
                    out.set(i, j, self.get(i, j) + other.get(0, j));
                }
            }
            return out;
        } else if (self.shape[0] == other.shape[0] and other.shape[1] == 1) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (0..self.shape[0]) |i| {
                for (0..self.shape[1]) |j| {
                    out.set(i, j, self.get(i, j) + other.get(i, 0));
                }
            }
            return out;
        }
        return error.ShapeMismatch;
    }

    pub fn mul(self: *const Tensor, other: *const Tensor) !Tensor {
        if (self.shape[0] == other.shape[0] and self.shape[1] == other.shape[1]) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (self.data, 0..) |v, i| {
                out.data[i] = v * other.data[i];
            }
            return out;
        } else if (self.shape[1] == other.shape[1] and other.shape[0] == 1) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (0..self.shape[0]) |i| {
                for (0..self.shape[1]) |j| {
                    out.set(i, j, self.get(i, j) * other.get(0, j));
                }
            }
            return out;
        } else if (self.shape[0] == other.shape[0] and other.shape[1] == 1) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (0..self.shape[0]) |i| {
                for (0..self.shape[1]) |j| {
                    out.set(i, j, self.get(i, j) * other.get(i, 0));
                }
            }
            return out;
        } else if (other.shape[0] == 1 and other.shape[1] == 1) {
            var out = try Tensor.init(self.allocator, self.shape);
            for (self.data, 0..) |v, i| {
                out.data[i] = v * other.data[0];
            }
            return out;
        }
        return error.ShapeMismatch;
    }

    pub fn matmul(self: *const Tensor, other: *const Tensor) !Tensor {
        std.debug.assert(self.shape[1] == other.shape[0]);
        var out = try Tensor.init(self.allocator, [2]usize{self.shape[0], other.shape[1]});
        
        for (0..self.shape[0]) |i| {
            for (0..other.shape[1]) |j| {
                var s: f32 = 0.0;
                for (0..self.shape[1]) |k| {
                    s += self.get(i, k) * other.get(k, j);
                }
                out.set(i, j, s);
            }
        }
        return out;
    }

    pub fn transpose(self: *const Tensor) !Tensor {
        var out = try Tensor.init(self.allocator, [2]usize{self.shape[1], self.shape[0]});
        for (0..self.shape[0]) |i| {
            for (0..self.shape[1]) |j| {
                out.set(j, i, self.get(i, j));
            }
        }
        return out;
    }

    pub fn sum(self: *const Tensor) f32 {
        var total: f32 = 0.0;
        for (self.data) |v| {
            total += v;
        }
        return total;
    }

    pub fn causal_mask(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        if (self.shape[0] != self.shape[1]) return error.ShapeMismatch;
        var res = try Tensor.init(allocator, self.shape);
        const n = self.shape[0];
        for (0..n) |i| {
            for (0..n) |j| {
                if (j > i) {
                    res.set(i, j, -1e9); // -inf approximation
                } else {
                    res.set(i, j, self.get(i, j));
                }
            }
        }
        return res;
    }
};

const testing = std.testing;

test "Tensor add" {
    const alloc = testing.allocator;
    var a = try Tensor.initData(alloc, .{2, 2}, &.{1.0, 2.0, 3.0, 4.0});
    defer a.deinit();
    var b = try Tensor.initData(alloc, .{2, 2}, &.{5.0, 6.0, 7.0, 8.0});
    defer b.deinit();
    
    var c = try a.add(&b);
    defer c.deinit();

    try testing.expectEqual(@as(f32, 6.0), c.get(0, 0));
    try testing.expectEqual(@as(f32, 8.0), c.get(0, 1));
    try testing.expectEqual(@as(f32, 10.0), c.get(1, 0));
    try testing.expectEqual(@as(f32, 12.0), c.get(1, 1));
}

test "Tensor matmul" {
    // [1 2]   [5 6]   [19 22]
    // [3 4] * [7 8] = [43 50]
    const alloc = testing.allocator;
    var a = try Tensor.initData(alloc, .{2, 2}, &.{1.0, 2.0, 3.0, 4.0});
    defer a.deinit();
    var b = try Tensor.initData(alloc, .{2, 2}, &.{5.0, 6.0, 7.0, 8.0});
    defer b.deinit();

    var c = try a.matmul(&b);
    defer c.deinit();

    try testing.expectEqual(@as(f32, 19.0), c.get(0, 0));
    try testing.expectEqual(@as(f32, 22.0), c.get(0, 1));
    try testing.expectEqual(@as(f32, 43.0), c.get(1, 0));
    try testing.expectEqual(@as(f32, 50.0), c.get(1, 1));
}

test "Tensor transpose" {
    const alloc = testing.allocator;
    var a = try Tensor.initData(alloc, .{2, 3}, &.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit();

    var b = try a.transpose();
    defer b.deinit();

    try testing.expectEqual(@as(usize, 3), b.shape[0]);
    try testing.expectEqual(@as(usize, 2), b.shape[1]);

    try testing.expectEqual(@as(f32, 1.0), b.get(0, 0));
    try testing.expectEqual(@as(f32, 4.0), b.get(0, 1));
    try testing.expectEqual(@as(f32, 2.0), b.get(1, 0));
    try testing.expectEqual(@as(f32, 5.0), b.get(1, 1));
    try testing.expectEqual(@as(f32, 3.0), b.get(2, 0));
    try testing.expectEqual(@as(f32, 6.0), b.get(2, 1));
}

test "Tensor sum" {
    const alloc = testing.allocator;
    var a = try Tensor.initData(alloc, .{2, 2}, &.{1.0, 2.0, 3.0, 4.0});
    defer a.deinit();
    
    try testing.expectEqual(@as(f32, 10.0), a.sum());
}
