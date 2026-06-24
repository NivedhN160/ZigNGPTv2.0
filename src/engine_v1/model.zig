const std = @import("std");
const markov = @import("markov.zig");

pub const TrigramModel = struct {
    map: markov.TrigramMap,

    pub fn init(allocator: std.mem.Allocator) !TrigramModel {
        return TrigramModel{ .map = markov.TrigramMap.init(allocator) };
    }

    pub fn deinit(self: *TrigramModel, allocator: std.mem.Allocator) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            e.value_ptr.deinit();
            allocator.free(e.key_ptr.a);
            allocator.free(e.key_ptr.b);
            allocator.free(e.key_ptr.c);
        }
        self.map.deinit();
    }

    pub fn generate(self: *@This(), allocator: std.mem.Allocator, len: usize, query: []const u8) ![]const u8 {
        if (self.map.count() == 0) return try allocator.dupe(u8, "Model is totally empty. Are you testing my patience?");
        var result = markov.ArrayList(u8).init(allocator);
        defer result.deinit();

        var query_words = markov.ArrayList([]const u8).init(allocator);
        defer query_words.deinit();
        var it = std.mem.tokenizeAny(u8, query, " \t\r\n,.!?");
        while (it.next()) |w| {
            try query_words.append(w);
        }

        var random = std.crypto.random;

        var keys = markov.ArrayList(markov.Trigram).init(allocator);
        defer keys.deinit();
        var kit = self.map.iterator();
        while (kit.next()) |e| {
            try keys.append(e.key_ptr.*);
        }

        var current_key = keys.items[random.intRangeLessThan(usize, 0, keys.items.len)];

        // Try to find a starting key matching user query
        if (query_words.items.len >= 1) {
            for (keys.items) |k| {
                if (std.ascii.eqlIgnoreCase(k.a, query_words.items[query_words.items.len-1])) {
                    current_key = k;
                    break;
                }
            }
        }

        try result.appendSlice(current_key.a);
        try result.append(' ');
        try result.appendSlice(current_key.b);
        try result.append(' ');
        try result.appendSlice(current_key.c);
        
        var words_gen: usize = 3;
        while (words_gen < len) : (words_gen += 1) {
            if (self.map.get(current_key)) |next_words| {
                if (next_words.items.len == 0) break;
                const next_word = next_words.items[random.intRangeLessThan(usize, 0, next_words.items.len)];
                try result.append(' ');
                try result.appendSlice(next_word);
                
                current_key = markov.Trigram{
                    .a = current_key.b,
                    .b = current_key.c,
                    .c = next_word,
                };
            } else {
                current_key = keys.items[random.intRangeLessThan(usize, 0, keys.items.len)];
            }
        }

        return try result.toOwnedSlice();
    }

        fn readU32(file: std.fs.File) !u32 {
        var buf: [4]u8 = undefined;
        _ = try file.readAll(&buf);
        return std.mem.readInt(u32, &buf, .little);
    }

    pub fn loadFromBinaryFile(self: *TrigramModel, filename: []const u8, allocator: std.mem.Allocator) !void {
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        const count = try readU32(file);
        try self.map.ensureTotalCapacity(count);

        var i: u32 = 0;
        while (i < count) : (i += 1) {
            const a_len = try readU32(file);
            const a = try allocator.alloc(u8, a_len);
            _ = try file.readAll(a);

            const b_len = try readU32(file);
            const b = try allocator.alloc(u8, b_len);
            _ = try file.readAll(b);

            const c_len = try readU32(file);
            const c = try allocator.alloc(u8, c_len);
            _ = try file.readAll(c);

            const key = markov.Trigram{ .a = a, .b = b, .c = c };
            var values = markov.ArrayList([]const u8).init(allocator);

            const val_count = try readU32(file);
            var j: u32 = 0;
            while (j < val_count) : (j += 1) {
                const w_len = try readU32(file);
                const w = try allocator.alloc(u8, w_len);
                _ = try file.readAll(w);
                try values.append(w);
            }

            try self.map.putNoClobber(key, values);
        }
    }
};






