const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tokenizer = struct {
    vocab: std.AutoHashMap(u8, usize),
    inverse_vocab: std.ArrayList(u8),

    pub fn init(allocator: Allocator, text: []const u8) !Tokenizer {
        var vocab = std.AutoHashMap(u8, usize).init(allocator);
        var inverse_vocab = std.ArrayList(u8).empty;

        var i: usize = 0;
        for (text) |char| {
            if (!vocab.contains(char)) {
                try vocab.put(char, i);
                try inverse_vocab.append(allocator, char);
                i += 1;
            }
        }

        return Tokenizer{
            .vocab = vocab,
            .inverse_vocab = inverse_vocab,
        };
    }

    pub fn deinit(self: *Tokenizer, allocator: Allocator) void {
        self.vocab.deinit();
        self.inverse_vocab.deinit(allocator);
    }

    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]usize {
        var indices = try allocator.alloc(usize, text.len);
        for (text, 0..) |char, i| {
            if (self.vocab.get(char)) |idx| {
                indices[i] = idx;
            } else {
                indices[i] = 0; // Fallback
            }
        }
        return indices;
    }

    pub fn decode(self: *const Tokenizer, allocator: Allocator, indices: []const usize) ![]u8 {
        var text = try allocator.alloc(u8, indices.len);
        for (indices, 0..) |idx, i| {
            if (idx < self.inverse_vocab.items.len) {
                text[i] = self.inverse_vocab.items[idx];
            } else {
                text[i] = '?'; // Fallback
            }
        }
        return text;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.inverse_vocab.items.len;
    }
};

test "Tokenizer encode and decode" {
    const allocator = std.testing.allocator;
    const text = "hello world";
    
    var tokenizer = try Tokenizer.init(allocator, text);
    defer tokenizer.deinit(allocator);

    const encoded = try tokenizer.encode(allocator, text);
    defer allocator.free(encoded);

    const decoded = try tokenizer.decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(text, decoded);
}
