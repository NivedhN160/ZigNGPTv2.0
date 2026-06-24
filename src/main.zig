const std = @import("std");
const cli = @import("cli.zig");

pub fn main() !void {
    try cli.main();
}