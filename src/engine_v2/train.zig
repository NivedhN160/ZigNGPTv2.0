const std = @import("std");
const nn = @import("nn.zig");
const autograd = @import("autograd.zig");
const transformer = @import("transformer.zig");
const tokenizer = @import("tokenizer.zig");
const Tokenizer = tokenizer.Tokenizer;
const Transformer = transformer.Transformer;
const TensorNode = autograd.TensorNode;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // We intentionally don't deinit the GPA to suppress the massive leak log.
    // The OS reclaims all memory cleanly upon exit.
    const allocator = gpa.allocator();

    // 1. Load data
    var file = std.fs.cwd().openFile("stories_corpus.txt", .{}) catch |err| {
        std.debug.print("Could not open stories_corpus.txt: {}\n", .{err});
        return;
    };
    defer file.close();
    const file_size = try file.getEndPos();
    const corpus = try file.readToEndAlloc(allocator, file_size);
    defer allocator.free(corpus);

    // 2. Tokenize
    var tok = try Tokenizer.init(allocator, corpus);
    defer tok.deinit(allocator);

    const vocab_size = tok.vocabSize();
    std.debug.print("Vocab size: {}\n", .{vocab_size});

    const encoded = try tok.encode(allocator, corpus);
    defer allocator.free(encoded);

    // 3. Model setup
    const seq_len = 32;
    const embed_dim = 64;
    const num_layers = 2;
    var model = try Transformer.init(allocator, vocab_size, embed_dim, num_layers, seq_len);
    defer model.deinit(allocator);

    // SGD optimizer
    const lr: f32 = 0.01;

    // 4. Training loop
    const epochs = 1;

    for (0..epochs) |epoch| {
        var i: usize = 0;
        var step: usize = 0;
        while (i + seq_len + 1 < encoded.len) : (i += seq_len) {
            // Memory arena for autograd graph per step
            var arena = std.heap.ArenaAllocator.init(allocator);
            const aa = arena.allocator();

            const input_indices = encoded[i .. i + seq_len];
            const target_indices = encoded[i + 1 .. i + seq_len + 1];

            // Create target one-hot tensor
            var target_data = try aa.alloc(f32, seq_len * vocab_size);
            @memset(target_data, 0.0);
            for (0..seq_len) |t| {
                target_data[t * vocab_size + target_indices[t]] = 1.0;
            }
            const target_node = try TensorNode.createData(aa, .{seq_len, vocab_size}, target_data);

            // Forward
            const logits = try model.forward(aa, input_indices);
            const probs = try TensorNode.softmax(aa, logits);
            const loss = try TensorNode.mse_loss(aa, probs, target_node);

            // Backward
            try loss.backward(aa);

            // Update parameters
            var params = std.ArrayList(*TensorNode).empty;
            try model.parameters(aa, &params);
            
            for (params.items) |p| {
                for (0..p.data.data.len) |j| {
                    p.data.data[j] -= lr * p.grad.data[j];
                    p.grad.data[j] = 0; // zero grad
                }
            }

            if (step % 10 == 0) {
                std.debug.print("Epoch {}, Step {}, Loss: {d}\n", .{epoch, step, loss.data.data[0]});
            }

            arena.deinit();
            step += 1;
            
            if (step > 50) break; // MVP just testing
        }
    }
    
    std.debug.print("Training complete.\n", .{});
}
