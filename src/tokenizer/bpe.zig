const std = @import("std");

pub const Tokenizer = struct {
    vocab: std.StringHashMap(u32),
    merges: std.ArrayList([2]u32),

    pub fn init(allocator: Allocator) !@This() {
        return .{
            .vocab = std.StringHashMap(u32).init(allocator),
            .merges = std.ArrayList([2]u32).init(allocator),
        };
    }

    pub fn encode(self: *@This(), text: []const u8) ![]u32 {
        // Implement BPE encoding (simplified)
        var tokens = std.ArrayList(u32).init(self.vocab.allocator);
        for (text) |char| {
            try tokens.append(@intCast(u32, char));
        }
        return tokens.toOwnedSlice();
    }

    pub fn decode(self: *@This(), tokens: []const u32) ![]u8 {
        var text = std.ArrayList(u8).init(self.vocab.allocator);
        for (tokens) |token| {
            try text.append(@intCast(u8, token));
        }
        return text.toOwnedSlice();
    }

    pub fn deinit(self: *@This()) void {
        self.vocab.deinit();
        self.merges.deinit();
    }
};