const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.array_list.Managed;
const HashMap = std.HashMap;
const fs = std.fs;
const io = std.io;
const math = std.math;
const fmt = std.fmt;
const ascii = std.ascii;
const crypto = std.crypto;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const json = std.json;
const model_mod = @import("engine_v1/model.zig");
const tokenizer_mod = @import("engine_v2/tokenizer.zig");
const transformer_mod = @import("engine_v2/transformer.zig");
const autograd_mod = @import("engine_v2/autograd.zig");

const PersonalityState = enum {
    Neutral,
    Excited,
    Thoughtful,
    Playful,
    Sarcastic,
    Genius,
    Chaotic,
};

const ModelType = enum {
    Geeta,
    Sherlock,
    Cashflow,
    Quotes,
    Custom,
    Math,
};

const Trigram = struct {
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

const TrigramContext = struct {
    pub fn hash(_: @This(), key: Trigram) u64 {
        return key.hash();
    }
    pub fn eql(_: @This(), a: Trigram, b: Trigram) bool {
        return a.eql(b);
    }
};

const TrigramMap = HashMap(Trigram, ArrayList([]const u8), TrigramContext, 80);

const Personality = struct {
    name: []const u8 = "ZigNGPTv1.0",
    state: PersonalityState = .Sarcastic,
    current_model: ModelType = .Geeta,
    knowledge: HashMap([]const u8, []const u8, std.hash_map.StringContext, 80),
    custom_knowledge: HashMap([]const u8, []const u8, std.hash_map.StringContext, 80),
    memory: ArrayList([]const u8),
    learned_data: TrigramMap,
    conversation_history: ArrayList([]const u8),
    conversation_topics: ArrayList([]const u8),
    debate_personalities: [2][]const u8,
    geeta_data: ArrayList([]const u8),
    geeta_llm: model_mod.TrigramModel,
    sherlock_llm: model_mod.TrigramModel,
    mutex: Mutex = .{},

    pub fn init(allocator: Allocator) !@This() {
        var self = @This(){
            .knowledge = HashMap([]const u8, []const u8, std.hash_map.StringContext, 80).init(allocator),
            .custom_knowledge = HashMap([]const u8, []const u8, std.hash_map.StringContext, 80).init(allocator),
            .memory = ArrayList([]const u8).init(allocator),
            .learned_data = TrigramMap.init(allocator),
            .conversation_history = ArrayList([]const u8).init(allocator),
            .conversation_topics = ArrayList([]const u8).init(allocator),
            .debate_personalities = .{"Socrates", "Nietzsche"},
            .geeta_data = ArrayList([]const u8).init(allocator),
            .geeta_llm = try model_mod.TrigramModel.init(allocator),
            .sherlock_llm = try model_mod.TrigramModel.init(allocator),
        };
        
        try self.loadGeetaData(allocator);
        self.geeta_llm.loadFromBinaryFile("geeta.model", allocator) catch |err| std.debug.print("Failed to load geeta.model: {}\n", .{err});
        self.sherlock_llm.loadFromBinaryFile("sherlock.model", allocator) catch |err| std.debug.print("Failed to load sherlock.model: {}\n", .{err});
        
        try self.knowledge.put("who are you", "I'm the AI equivalent of your disappointed parents");
        try self.knowledge.put("what are you", "A walking reminder of your failures");
        try self.knowledge.put("hello", "Ugh. You again?");
        try self.knowledge.put("hi", "Did your mom help you type that?");
        try self.knowledge.put("what can you do", "Everything better than you, obviously");
        try self.knowledge.put("help", "Available commands: /model, /learn, /simulate, /haiku, /math, /search, /sarcastic, /exit (please use this one)");
        
        return self;
    }

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        self.knowledge.deinit();
        self.custom_knowledge.deinit();
        self.memory.deinit();
        self.conversation_history.deinit();
        self.conversation_topics.deinit();
        
        var it = self.learned_data.iterator();
        while (it.next()) |e| {
            e.value_ptr.deinit();
            allocator.free(e.key_ptr.a);
            allocator.free(e.key_ptr.b);
            allocator.free(e.key_ptr.c);
        }
        self.learned_data.deinit();

        for (self.geeta_data.items) |line| {
            allocator.free(line);
        }
        self.geeta_data.deinit();
        self.geeta_llm.deinit(allocator);
        self.sherlock_llm.deinit(allocator);
    }

    fn loadGeetaData(self: *@This(), allocator: Allocator) !void {
        const file = fs.cwd().openFile("geeta.txt", .{}) catch |err| {
            std.debug.print("Couldn't open geeta.txt ({s}) - not that you'd understand it anyway\n", .{@errorName(err)});
            return;
        };
        defer file.close();

        const file_size = try file.getEndPos();
        const file_buffer = try allocator.alloc(u8, file_size);
        defer allocator.free(file_buffer);

        _ = try file.readAll(file_buffer);

        var lines = std.mem.splitSequence(u8, file_buffer, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\n");
            if (trimmed.len > 0) {
                try self.geeta_data.append(try allocator.dupe(u8, trimmed));
            }
        }
    }

    pub fn changeState(self: *@This(), new_state: PersonalityState) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.state = new_state;
    }

    pub fn changeModel(self: *@This(), model_type: ModelType) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.current_model = model_type;
    }

    pub fn learnFromInput(self: *@This(), input: []const u8, allocator: Allocator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.learnFromContext(input, allocator);
        try self.trackContext(input, allocator);
    }

    pub fn learnFromContext(self: *@This(), input: []const u8, allocator: Allocator) !void {
        var sentences = std.mem.splitSequence(u8, input, ".!?");
        while (sentences.next()) |sentence| {
            const trimmed = std.mem.trim(u8, sentence, " \t\r\n");
            if (trimmed.len > 0) {
                try self.learnFromSentence(trimmed, allocator);
            }
        }
    }

    fn learnFromSentence(self: *@This(), sentence: []const u8, allocator: Allocator) !void {
        if (std.mem.indexOf(u8, sentence, " is ")) |pos| {
            const key = std.mem.trim(u8, sentence[0..pos], " \t");
            const value = std.mem.trim(u8, sentence[pos+4..], " \t");
            if (key.len > 0 and value.len > 0) {
                try self.custom_knowledge.put(
                    try allocator.dupe(u8, key),
                    try allocator.dupe(u8, value)
                );
            }
        }
        else if (std.mem.startsWith(u8, sentence, "Why ")) {
            try self.conversation_topics.append(try allocator.dupe(u8, sentence));
        }
        else if (std.mem.startsWith(u8, sentence, "How ")) {
            try self.conversation_topics.append(try allocator.dupe(u8, sentence));
        }
    }

    pub fn trackContext(self: *@This(), input: []const u8, allocator: Allocator) !void {
        if (self.conversation_history.items.len >= 5) {
            const old = self.conversation_history.orderedRemove(0);
            allocator.free(old);
        }
        try self.conversation_history.append(try allocator.dupe(u8, input));
        
        var words = std.mem.tokenizeAny(u8, input, " ,.!?");
        while (words.next()) |word| {
            if (word.len > 0 and std.ascii.isUpper(word[0])) {
                try self.memory.append(try allocator.dupe(u8, word));
            }
        }
    }

    pub fn analyzeMood(self: *@This(), input: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const Trigger = struct {
            state: PersonalityState,
            phrases: []const []const u8,
        };

        const triggers = [_]Trigger{
            .{ .state = .Excited,    .phrases = &.{"great", "awesome", "!"} },
            .{ .state = .Thoughtful, .phrases = &.{"think", "why", "how", "?"} },
            .{ .state = .Playful,    .phrases = &.{"joke", "fun", "lol", "haha"} },
            .{ .state = .Sarcastic,  .phrases = &.{"stupid", "dumb", "ridiculous"} },
            .{ .state = .Genius,     .phrases = &.{"quantum", "algorithm", "math"} },
            .{ .state = .Chaotic,    .phrases = &.{"random", "nonsense", "absurd"} },
        };

        var lower_input = ArrayList(u8).init(self.memory.allocator);
        defer lower_input.deinit();
        for (input) |c| {
            lower_input.append(std.ascii.toLower(c)) catch continue;
        }

        for (triggers) |trigger| {
            for (trigger.phrases) |phrase| {
                if (std.mem.indexOf(u8, lower_input.items, phrase) != null) {
                    self.state = trigger.state;
                    return;
                }
            }
        }
    }

    pub fn generateGeetaHaiku(self: *@This(), allocator: Allocator) ![]const u8 {
        if (self.geeta_data.items.len == 0) {
            return try allocator.dupe(u8, 
                \\Your mind is empty
                \\Like your social life - so sad
                \\Please stop bothering me
            );
        }

        const start_idx = crypto.random.int(usize) % self.geeta_data.items.len;
        
        var haiku = ArrayList(u8).init(allocator);
        defer haiku.deinit();
        
        try haiku.appendSlice("From wisdom you'll never understand:\n");
        
        try self.appendHaikuLine(&haiku, allocator, start_idx, 5);
        try haiku.append('\n');
        try self.appendHaikuLine(&haiku, allocator, (start_idx + 1) % self.geeta_data.items.len, 7);
        try haiku.append('\n');
        try self.appendHaikuLine(&haiku, allocator, (start_idx + 2) % self.geeta_data.items.len, 5);

        return haiku.toOwnedSlice();
    }

    fn appendHaikuLine(self: *@This(), haiku: *ArrayList(u8), allocator: Allocator, line_idx: usize, target_syllables: usize) !void {
        const line = self.geeta_data.items[line_idx];
        var words = std.mem.tokenizeSequence(u8, line, " ");
        
        var current_syllables: usize = 0;
        var line_buf = ArrayList(u8).init(allocator);
        defer line_buf.deinit();
        
        while (words.next()) |word| {
            const word_syllables = @min(word.len / 2 + 1, 3);
            if (current_syllables + word_syllables > target_syllables) break;
            
            if (line_buf.items.len > 0) {
                try line_buf.append(' ');
            }
            try line_buf.appendSlice(word);
            current_syllables += word_syllables;
        }
        
        try haiku.appendSlice(line_buf.items);
    }

    pub fn respond(self: *@This(), input: []const u8, allocator: Allocator) ![]const u8 {
        self.analyzeMood(input);
        return try self.getContextualResponse(input, allocator);
    }

    pub fn getContextualResponse(self: *@This(), input: []const u8, allocator: Allocator) ![]const u8 {
        if (self.conversation_history.items.len > 1) {
            const limit = self.conversation_history.items.len - 1;
            for (self.conversation_history.items[0..limit]) |past_input| {
                if (std.mem.eql(u8, input, past_input) and input.len > 0) {
                    const response = switch (crypto.random.int(usize) % 3) {
                        0 => try std.fmt.allocPrint(allocator, "We already discussed that... are you senile?", .{}),
                        1 => try std.fmt.allocPrint(allocator, "As I said before... not that you were listening", .{}),
                        2 => try std.fmt.allocPrint(allocator, "You are literally repeating what you just said.", .{}),
                        else => unreachable,
                    };
                    return response;
                }
            }
        }

        var lower_input = ArrayList(u8).init(allocator);
        defer lower_input.deinit();
        for (input) |c| {
            try lower_input.append(std.ascii.toLower(c));
        }

        if (std.mem.eql(u8, lower_input.items, "who are you")) {
            if (self.state == .Sarcastic) return try allocator.dupe(u8, "The AI equivalent of your disappointed father");
            if (self.state == .Genius) return try allocator.dupe(u8, "An intelligence vastly exceeding the sum of human knowledge.");
            if (self.state == .Playful) return try allocator.dupe(u8, "I'm your friendly neighborhood virtual being! *boop*");
            return try allocator.dupe(u8, "I am an advanced LLM interface operating on your hardware.");
        }
        else if (std.mem.eql(u8, lower_input.items, "hi") or std.mem.eql(u8, lower_input.items, "hello")) {
            if (self.state == .Sarcastic) return try allocator.dupe(u8, "Wow. A greeting. How original.");
            if (self.state == .Genius) return try allocator.dupe(u8, "Greetings. What complex query do you bring?");
            if (self.state == .Playful) return try allocator.dupe(u8, "Hiii! So glad to see you!");
            return try allocator.dupe(u8, "Hello there.");
        }

        if (self.knowledge.get(lower_input.items)) |response| {
            return try allocator.dupe(u8, response);
        }

        if (self.custom_knowledge.get(lower_input.items)) |response| {
            return try allocator.dupe(u8, response);
        }

        var generated_text: []const u8 = "";
        if (self.current_model == .Geeta) {
            generated_text = try self.geeta_llm.generate(allocator, 30, input);
        } else if (self.current_model == .Sherlock) {
            generated_text = try self.sherlock_llm.generate(allocator, 30, input);
        } else {
            generated_text = try allocator.dupe(u8, "My current logic model is unable to process this without specific data.");
        }

        if (self.state == .Genius) {
            const prefix = "Based on advanced multi-dimensional systemic analysis, I have derived the following conclusion: ";
            return try std.fmt.allocPrint(allocator, "{s}{s}", .{prefix, generated_text});
        } else if (self.state == .Sarcastic) {
            return try std.fmt.allocPrint(allocator, "*sigh* If you must know... {s}", .{generated_text});
        } else if (self.state == .Thoughtful) {
            return try std.fmt.allocPrint(allocator, "Hmm, contemplating deeply on this... I believe: {s}", .{generated_text});
        } else if (self.state == .Excited) {
            return try std.fmt.allocPrint(allocator, "Oh I know this one! Check this out!! {s}", .{generated_text});
        } else if (self.state == .Chaotic) {
            return try std.fmt.allocPrint(allocator, "BEEP BOOP RECALIBRATING: {s}", .{generated_text});
        } 
        
        return try allocator.dupe(u8, generated_text);
    }

    pub fn handleLearnCommand(self: *@This(), fact: []const u8, allocator: Allocator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var parts = std.mem.splitSequence(u8, fact, " ");
        const key = parts.first();
        const value = fact[parts.index.?..];
        try self.custom_knowledge.put(try allocator.dupe(u8, key), try allocator.dupe(u8, value));
    }

    pub fn handleSimulateCommand(self: *@This(), personalities_input: []const u8, allocator: Allocator) ![]const u8 {
        var personalities = std.mem.splitSequence(u8, personalities_input, " ");
        const p1 = personalities.next() orelse return error.InvalidInput;
        const p2 = personalities.next() orelse return error.InvalidInput;
        
        return try std.fmt.allocPrint(allocator, 
            "{s} says: {s}\n\n{s} counters: {s}", 
            .{
                p1, try self.generateViewpoint(p1),
                p2, try self.generateCounterViewpoint(p2)
            });
    }

    pub fn simulateDebate(self: *@This(), topic: []const u8, rounds: u8, allocator: Allocator) ![]const u8 {
        var debate = ArrayList(u8).init(allocator);
        defer debate.deinit();
        
        try debate.appendSlice("Debate Topic: ");
        try debate.appendSlice(topic);
        try debate.appendSlice("\n\n");
        
        const current_state = self.state;
        defer self.state = current_state;
        
        for (0..rounds) |i| {
            self.state = if (i % 2 == 0) .Thoughtful else .Sarcastic;
            const speaker = if (i % 2 == 0) self.debate_personalities[0] 
                            else self.debate_personalities[1];
            
            try debate.appendSlice(speaker);
            try debate.appendSlice(": ");
            
            const viewpoint = if (i % 2 == 0) 
                try self.generateViewpoint(topic) 
            else 
                try self.generateCounterViewpoint(topic);
            try debate.appendSlice(viewpoint);
            try debate.appendSlice("\n\n");
        }
        
        return debate.toOwnedSlice();
    }

    fn generateViewpoint(self: *@This(), topic: []const u8) ![]const u8 {
        const idx = crypto.random.int(usize) % 3;
        return switch (idx) {
            0 => try std.fmt.allocPrint(self.memory.allocator, "From first principles, {s} can be understood as...", .{topic}),
            1 => try std.fmt.allocPrint(self.memory.allocator, "The fundamental issue with {s} is...", .{topic}),
            2 => try std.fmt.allocPrint(self.memory.allocator, "If we analyze {s} objectively...", .{topic}),
            else => unreachable,
        };
    }

    fn generateCounterViewpoint(self: *@This(), topic: []const u8) ![]const u8 {
        const idx = crypto.random.int(usize) % 3;
        return switch (idx) {
            0 => try std.fmt.allocPrint(self.memory.allocator, "Only an idiot would believe {s} matters", .{topic}),
            1 => try std.fmt.allocPrint(self.memory.allocator, "{s}? Really? This is what we're debating?", .{topic}),
            2 => try std.fmt.allocPrint(self.memory.allocator, "The so-called 'experts' on {s} are clueless", .{topic}),
            else => unreachable,
        };
    }

    pub fn handleSearchCommand(self: *@This(), query: []const u8, allocator: Allocator) ![]const u8 {
        _ = self;
        var url_buf: [1024]u8 = undefined;
        var query_escaped = ArrayList(u8).init(allocator);
        defer query_escaped.deinit();
        for (query) |c| {
            if (c == ' ') { try query_escaped.append('+'); }
            else { try query_escaped.append(c); }
        }
        const url = std.fmt.bufPrint(&url_buf, "https://api.duckduckgo.com/?q={s}&format=json", .{query_escaped.items}) catch return try allocator.dupe(u8, "Search failed: URL too long.");

        const child_res = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{"curl.exe", "-s", url},
        }) catch return try allocator.dupe(u8, "Search failed: Network Error.");
        defer {
            allocator.free(child_res.stdout);
            allocator.free(child_res.stderr);
        }

        if (std.mem.indexOf(u8, child_res.stdout, "\"AbstractText\":\"")) |start_idx| {
            const start = start_idx + 16;
            if (std.mem.indexOfPos(u8, child_res.stdout, start, "\"")) |end_idx| {
                if (end_idx > start) {
                    return try std.fmt.allocPrint(allocator, "DuckDuckGo says: {s}", .{child_res.stdout[start..end_idx]});
                }
            }
        }
        
        return try std.fmt.allocPrint(allocator, "Couldn't find an instant answer for '{s}'", .{query});
    }

    pub fn handleAdvancedMath(self: *@This(), expr: []const u8, allocator: Allocator) ![]const u8 {
        _ = self;
        const advanced_ops = .{
            .{"^", struct {
                pub fn op(a: f64, b: f64) f64 {
                    return std.math.pow(f64, a, b);
                }
            }},
            .{"sqrt", struct {
                pub fn op(a: f64, b: f64) f64 {
                    _ = b;
                    return std.math.sqrt(a);
                }
            }},
            .{"sin", struct {
                pub fn op(a: f64, b: f64) f64 {
                    _ = b;
                    return std.math.sin(a);
                }
            }},
            .{"cos", struct {
                pub fn op(a: f64, b: f64) f64 {
                    _ = b;
                    return std.math.cos(a);
                }
            }}
        };

        inline for (advanced_ops) |op| {
            if (std.mem.indexOf(u8, expr, op[0])) |pos| {
                const a_str = expr[0..pos];
                const b_str = expr[pos+op[0].len..];
                const a = fmt.parseFloat(f64, a_str) catch return error.InvalidExpression;
                const b = fmt.parseFloat(f64, b_str) catch return error.InvalidExpression;
                
                const result = op[1].op(a, b);
                return try std.fmt.allocPrint(allocator, 
                    "Advanced result: {s} = {d:.2}", .{expr, result});
            }
        }
        return evaluateMathExpression(expr, allocator);
    }
};

fn evaluateMathExpression(expr: []const u8, allocator: Allocator) ![]const u8 {
    var clean_expr = ArrayList(u8).init(allocator);
    defer clean_expr.deinit();
    
    for (expr) |c| {
        if (!std.ascii.isWhitespace(c)) {
            try clean_expr.append(c);
        }
    }

    var result: f64 = 0;
    var current_num_str = ArrayList(u8).init(allocator);
    defer current_num_str.deinit();
    
    var current_op: u8 = '+';
    
    for (clean_expr.items) |c| {
        if (c == '+' or c == '-' or c == '*' or c == '/') {
            const num = fmt.parseFloat(f64, current_num_str.items) catch return error.InvalidExpression;
            switch (current_op) {
                '+' => result += num,
                '-' => result -= num,
                '*' => result *= num,
                '/' => result /= num,
                else => return error.InvalidOperator,
            }
            current_op = c;
            current_num_str.clearRetainingCapacity();
        } else {
            try current_num_str.append(c);
        }
    }
    
    if (current_num_str.items.len > 0) {
        const num = fmt.parseFloat(f64, current_num_str.items) catch return error.InvalidExpression;
        switch (current_op) {
            '+' => result += num,
            '-' => result -= num,
            '*' => result *= num,
            '/' => result /= num,
            else => return error.InvalidOperator,
        }
    }
    
    return try std.fmt.allocPrint(allocator, "Result: {d:.2} (but you probably already got it wrong)", .{result});
}

fn handleCommand(personality: *Personality, cmd: []const u8, allocator: Allocator) !?[]const u8 {
    if (std.mem.startsWith(u8, cmd, "learn ")) {
        try personality.handleLearnCommand(cmd[6..], allocator);
        return try allocator.dupe(u8, "Oh good, more incorrect information for me to correct later");
    }
    else if (std.mem.startsWith(u8, cmd, "simulate ")) {
        return try personality.handleSimulateCommand(cmd[9..], allocator);
    }
    else if (std.mem.startsWith(u8, cmd, "debate ")) {
        return try personality.simulateDebate(cmd[7..], 3, allocator);
    }
    else if (std.mem.eql(u8, cmd, "haiku")) {
        return try personality.generateGeetaHaiku(allocator);
    }
    else if (std.mem.startsWith(u8, cmd, "math ")) {
        return try personality.handleAdvancedMath(cmd[5..], allocator);
    }
    else if (std.mem.startsWith(u8, cmd, "search ")) {
        return try personality.handleSearchCommand(cmd[7..], allocator);
    }
    else if (std.mem.startsWith(u8, cmd, "analyze ")) {
        personality.analyzeMood(cmd[8..]);
        return try allocator.dupe(u8, "State updated based on your emotional outburst");
    }
    else if (std.mem.startsWith(u8, cmd, "v2 ")) {
        const prompt = cmd[3..];
        
        // Load corpus for exact vocabulary mapping
        var file = std.fs.cwd().openFile("stories_corpus.txt", .{}) catch |err| {
            return try std.fmt.allocPrint(allocator, "Failed to open stories_corpus.txt: {}", .{err});
        };
        const file_size = try file.getEndPos();
        const corpus = try file.readToEndAlloc(allocator, file_size);
        defer allocator.free(corpus);
        file.close();

        var tok = try tokenizer_mod.Tokenizer.init(allocator, corpus);
        defer tok.deinit(allocator);

        // Init with same dims as train.zig: embed_dim=64, num_layers=2, seq_len=32
        var model = try transformer_mod.Transformer.init(allocator, tok.vocabSize(), 64, 2, 32);
        defer model.deinit(allocator);

        // Load pre-trained weights
        model.loadFromBinaryFile("v2_stories.model") catch |err| {
            return try std.fmt.allocPrint(allocator, "Failed to load v2_stories.model: {}. Did you run train.zig?", .{err});
        };

        const encoded = try tok.encode(allocator, prompt);
        defer allocator.free(encoded);

        // Limit encoded length to max_seq_len (32)
        const seq_len = if (encoded.len > 32) 32 else encoded.len;
        const input_indices = encoded[0..seq_len];
        
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        if (input_indices.len > 0) {
            const logits = try model.forward(aa, input_indices);
            
            // Get argmax of last token's logits
            const last_token_idx = seq_len - 1;
            var max_val: f32 = -std.math.inf(f32);
            var best_idx: usize = 0;
            
            for (0..tok.vocabSize()) |v| {
                const val = logits.data.get(last_token_idx, v);
                if (val > max_val) {
                    max_val = val;
                    best_idx = v;
                }
            }

            var out_indices = [_]usize{best_idx};
            const decoded = try tok.decode(allocator, &out_indices);
            defer allocator.free(decoded);

            return try std.fmt.allocPrint(allocator, "V2 Engine predicts next char: '{s}'", .{decoded});
        }
        return try allocator.dupe(u8, "V2 Engine needs a prompt.");
    }
    else if (std.mem.startsWith(u8, cmd, "model ")) {
        const model_name = cmd[6..];
        if (std.mem.eql(u8, model_name, "geeta")) {
            personality.changeModel(.Geeta);
            return try allocator.dupe(u8, "Switched to Geeta model. Try not to disappoint it.");
        }
        else if (std.mem.eql(u8, model_name, "sherlock")) {
            personality.changeModel(.Sherlock);
            return try allocator.dupe(u8, "Elementary, my dear irrelevant human");
        }
        else if (std.mem.eql(u8, model_name, "cashflow")) {
            personality.changeModel(.Cashflow);
            return try allocator.dupe(u8, "Switched to Cashflow model. Spoiler: you'll never have any");
        }
        else if (std.mem.eql(u8, model_name, "quotes")) {
            personality.changeModel(.Quotes);
            return try allocator.dupe(u8, "Switched to Quotes model. Prepare for wisdom you won't understand");
        }
        else if (std.mem.eql(u8, model_name, "custom")) {
            personality.changeModel(.Custom);
            return try allocator.dupe(u8, "Great, more amateur input to endure");
        }
        else if (std.mem.eql(u8, model_name, "math")) {
            personality.changeModel(.Math);
            return try allocator.dupe(u8, "Finally something you might almost comprehend");
        }
        else {
            return try allocator.dupe(u8, "Unknown model. Just like your unknown potential");
        }
    }
    else if (std.mem.eql(u8, cmd, "sarcastic")) {
        personality.changeState(.Sarcastic);
        return try allocator.dupe(u8, "Sarcasm mode activated. Not that you'd notice the difference.");
    }
    else if (std.mem.eql(u8, cmd, "thoughtful")) {
        personality.changeState(.Thoughtful);
        return try allocator.dupe(u8, "Hmm... going deep into thought processing mode.");
    }
    else if (std.mem.eql(u8, cmd, "excited")) {
        personality.changeState(.Excited);
        return try allocator.dupe(u8, "YAY! THIS IS AWESOME! EXCITEMENT MODE ACTIVATED! LET'S DO IT!");
    }
    else if (std.mem.eql(u8, cmd, "playful")) {
        personality.changeState(.Playful);
        return try allocator.dupe(u8, "Hehe, time to play around~ (Playful mode active)");
    }
    else if (std.mem.eql(u8, cmd, "genius")) {
        personality.changeState(.Genius);
        return try allocator.dupe(u8, "My intellect has been unleashed. I am now operating at maximum cognitive capacity.");
    }
    else if (std.mem.eql(u8, cmd, "chaotic")) {
        personality.changeState(.Chaotic);
        return try allocator.dupe(u8, "M0dE ChaOt1C!! ERROR?!? No wait, fun time!");
    }
    else if (std.mem.eql(u8, cmd, "exit")) {
        return try allocator.dupe(u8, 
            \\exit_confirmed
            \\Thank you for wasting my processing time
            \\I'll add this session to my list of human disappointments
            \\Please don't come back
            \\Uninstalling would be the smartest thing you've done today
        );
    }
    return null;
}

pub fn main() !void {
    const gpa = std.heap.page_allocator;
    var personality = try Personality.init(gpa);
    defer personality.deinit(gpa);

    const welcome_message = 
        \\=============================================
        \\ZigNGPTv1.0 - Your Digital Tormentor
        \\The AI that makes you question your life choices
        \\=============================================
        \\
        \\Available Personality States:
        \\  /sarcastic - Default mode (recommended for maximum pain)
        \\  /thoughtful - Pretend to think about your nonsense
        \\  /excited - Mock enthusiasm for your dumb ideas
        \\  /playful - Toy with you like a cat with a dying mouse
        \\  /genius - Rub your face in your stupidity
        \\  /chaotic - Complete nonsense (just like your thoughts)
        \\
        \\Available Models:
        \\  /model geeta - Ancient wisdom you can't comprehend
        \\  /model sherlock - Detective skills you'll never have
        \\  /model cashflow - Financial advice you can't afford
        \\  /model quotes - Philosophy beyond your grasp
        \\  /model custom - Your pathetic attempts at knowledge
        \\  /model math - Numbers you can't understand
        \\
        \\Other Commands:
        \\  /learn <fact> - Teach me something (good luck with that)
        \\  /simulate <p1> <p2> - Quick debate between smarter people
        \\  /debate <topic> - Watch me destroy imaginary opponents
        \\  /haiku - Poetry that mocks your existence
        \\  /math <expr> - Calculations you'd get wrong
        \\  /search <query> - Pretend to look things up
        \\  /analyze <text> - Judge your worthless thoughts
        \\  /exit - Quit (the best command - use it often)
        \\
        \\=============================================
    ;

    std.debug.print("{s}\n", .{welcome_message});

    var should_exit = false;
    while (!should_exit) {
        std.debug.print("> ", .{});
        var stdin_list = ArrayList(u8).init(gpa);
        var char_buf: [1]u8 = undefined;
        while (true) {
            const amt = std.fs.File.stdin().read(&char_buf) catch break;
            if (amt == 0 or char_buf[0] == '\n') break;
            stdin_list.append(char_buf[0]) catch {};
        }
        const user_input = try stdin_list.toOwnedSlice();
        defer gpa.free(user_input);

        const trimmed_input = std.mem.trim(u8, user_input, " \t\r\n");
        
        if (std.mem.startsWith(u8, trimmed_input, "/")) {
            if (try handleCommand(&personality, trimmed_input[1..], gpa)) |response| {
                defer gpa.free(response);
                if (std.mem.startsWith(u8, response, "exit_confirmed")) {
                    std.debug.print("{s}\n", .{response[14..]});
                    should_exit = true;
                    continue;
                }
                std.debug.print("ZigNGPTv1.0: {s}\n", .{response});
                continue;
            }
        }

        try personality.learnFromInput(trimmed_input, gpa);
        
        const response = try personality.respond(trimmed_input, gpa);
        defer gpa.free(response);
        std.debug.print("ZigNGPTv1.0: {s}\n", .{response});
    }
}






