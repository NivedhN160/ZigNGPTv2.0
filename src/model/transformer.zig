const std = @import("std");

pub const Transformer = struct {
    allocator: std.mem.Allocator,
    conversation_id: u32,
    knowledge: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator) !@This() {
        var self = @This(){
            .allocator = allocator,
            .conversation_id = std.crypto.random.int(u32),
            .knowledge = std.StringHashMap([]const u8).init(allocator),
        };

        // Basic knowledge base
        try self.knowledge.put("who are you", "I'm ZigNGPT, your Zig-powered assistant");
        try self.knowledge.put("what is zig", "Zig is a general-purpose programming language");
        try self.knowledge.put("hi", "Hello there! How can I help?");
        
        return self;
    }

    pub fn respond(self: *@This(), input: []const u8) ![]const u8 {
        // Check knowledge base first
        if (self.knowledge.get(input)) |response| {
            return try std.fmt.allocPrint(
                self.allocator,
                "[{x}] {s}",
                .{self.conversation_id, response}
            );
        }

        // Default response
        return try std.fmt.allocPrint(
            self.allocator,
            "[{x}] I'm not sure about '{s}'. Can you rephrase?",
            .{self.conversation_id, input}
        );
    }

    pub fn deinit(self: *@This()) void {
        self.knowledge.deinit();
    }
};