const std = @import("std"); pub fn main() !void { const f = try std.fs.cwd().openFile("model.zig", .{}); var buf: [4]u8 = undefined; _ = try f.readAll(&buf); }
