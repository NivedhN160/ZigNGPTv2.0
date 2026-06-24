const std = @import("std");

pub const Xoroshiro128Plus = struct {
    s0: u64,
    s1: u64,

    pub fn init(seed: u64) Xoroshiro128Plus {
        const s1 = seed ^ 0x9E3779B97F4A7C15;
        const s0 = s1 ^ 0xBF58476D1CE4E5B9;
        return .{ .s0 = s0, .s1 = s1 };
    }

    pub fn next(self: *Xoroshiro128Plus) u64 {
        var s0 = self.s0;
        var s1 = self.s1;
        const result = s0 + s1;

        s1 ^= s0;
        self.s0 = std.math.rotl(s0, 55) ^ s1 ^ (s1 << 14);
        self.s1 = std.math.rotl(s1, 36);
        return result;
    }

    pub fn intRangeLessThan(self: *Xoroshiro128Plus, comptime T: type, min: T, span: T) T {
        return @as(T, @intCast(self.next() % @as(u64, span))) + min;
    }
};
