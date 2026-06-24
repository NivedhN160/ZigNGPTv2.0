const std = @import("std");
const Allocator = std.mem.Allocator;
pub const ArrayList = std.array_list.Managed;
const HashMap = std.HashMap;

pub const Trigram = struct {
    a: []const u8,
    b: []const u8,
    c: []const u8,

    pub fn hash(self: @This()) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(self.a);
        h.update(self.b);
        h.update(self.c);
        return h.final();
    }

    pub fn eql(self: @This(), other: @This()) bool {
        return std.mem.eql(u8, self.a, other.a) and
               std.mem.eql(u8, self.b, other.b) and
               std.mem.eql(u8, self.c, other.c);
    }
};

pub const TrigramContext = struct {
    pub fn hash(_: @This(), key: Trigram) u64 {
        return key.hash();
    }

    pub fn eql(_: @This(), a: Trigram, b: Trigram) bool {
        return a.eql(b);
    }
};

pub const TrigramMap = HashMap(Trigram, ArrayList([]const u8), TrigramContext, 80);

const Personality = struct {
    name: []const u8 = "NGPT",
    knowledge: HashMap([]const u8, []const u8, std.hash_map.StringContext, 80),
    memory: HashMap([]const u8, []const u8, std.hash_map.StringContext, 80),
    conversation_history: ArrayList([]const u8),
    
    pub fn init(allocator: Allocator) @This() {
        return .{
            .knowledge = HashMap([]const u8, []const u8, std.hash_map.StringContext, 80).init(allocator),
            .memory = HashMap([]const u8, []const u8, std.hash_map.StringContext, 80).init(allocator),
            .conversation_history = ArrayList([]const u8).init(allocator),
        };
    }
};

fn trainModel(allocator: Allocator, input_file: []const u8) !TrigramMap {
    const file = try std.fs.cwd().openFile(input_file, .{ .mode = .read_only });
    defer file.close();

    const file_size = (try file.stat()).size;
    const content = try file.readToEndAlloc(allocator, file_size);
    defer allocator.free(content);

    var tokens = std.mem.tokenizeAny(u8, content, " \t\n\r");
    var words = ArrayList([]const u8).init(allocator);
    defer {
        for (words.items) |word| allocator.free(word);
        words.deinit();
    }

    while (tokens.next()) |token| {
        const word = try allocator.dupe(u8, token);
        try words.append(word);
    }

    if (words.items.len < 3) {
        std.debug.print("Input file is too small to generate trigrams\n", .{});
        return error.NotEnoughData;
    }

    var model = TrigramMap.init(allocator);
    errdefer {
        var it = model.iterator();
        while (it.next()) |e| {
            e.value_ptr.deinit();
            allocator.free(e.key_ptr.a);
            allocator.free(e.key_ptr.b);
            allocator.free(e.key_ptr.c);
        }
        model.deinit();
    }

    for (0..words.items.len - 2) |i| {
        const a = words.items[i];
        const b = words.items[i + 1];
        const c = words.items[i + 2];

        const key = Trigram{
            .a = try allocator.dupe(u8, a),
            .b = try allocator.dupe(u8, b),
            .c = try allocator.dupe(u8, c),
        };

        const gop = try model.getOrPut(key);
        if (!gop.found_existing) {
            gop.key_ptr.* = key;
            gop.value_ptr.* = ArrayList([]const u8).init(allocator);
        }

        if (i + 3 < words.items.len) {
            const next_word = words.items[i + 3];
            try gop.value_ptr.append(try allocator.dupe(u8, next_word));
        }
    }

    return model;
}

fn saveModel(model: TrigramMap, output_file: []const u8) !void {
    const out_file = try std.fs.cwd().createFile(output_file, .{});
    defer out_file.close();
    var writer_buf: [4096]u8 = undefined; const writer = out_file.writer(&writer_buf);

    try writer.writeInt(u32, @as(u32, @intCast(model.count())), .little);

    var it = model.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        try writer.writeInt(u32, @as(u32, @intCast(key.a.len)), .little);
        try writer.writeAll(key.a);
        try writer.writeInt(u32, @as(u32, @intCast(key.b.len)), .little);
        try writer.writeAll(key.b);
        try writer.writeInt(u32, @as(u32, @intCast(key.c.len)), .little);
        try writer.writeAll(key.c);

        const values = entry.value_ptr.*;
        try writer.writeInt(u32, @as(u32, @intCast(values.items.len)), .little);
        for (values.items) |word| {
            try writer.writeInt(u32, @as(u32, @intCast(word.len)), .little);
            try writer.writeAll(word);
        }
    }

    std.debug.print("Model saved to {s}\n", .{output_file});
}

pub fn main() !void {
    const gpa = std.heap.page_allocator;
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <input_file> [--save output.model]\n", .{args[0]});
        std.debug.print("If --save is omitted, will just build the model in memory\n", .{});
        return;
    }

    const input_file = args[1];
    var output_file: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--save") and i + 1 < args.len) {
            output_file = args[i + 1];
            i += 1;
        }
    }

    std.debug.print("Building trigram model from {s}...\n", .{input_file});
    var model = try trainModel(gpa, input_file);
    defer {
        var it = model.iterator();
        while (it.next()) |e| {
            e.value_ptr.deinit();
            gpa.free(e.key_ptr.a);
            gpa.free(e.key_ptr.b);
            gpa.free(e.key_ptr.c);
        }
        model.deinit();
    }

    std.debug.print("Model built with {} trigrams\n", .{model.count()});

    if (output_file) |out_file| {
        try saveModel(model, out_file);
    } else {
        std.debug.print("Model not saved (use --save filename.model to save)\n", .{});
    }
}


