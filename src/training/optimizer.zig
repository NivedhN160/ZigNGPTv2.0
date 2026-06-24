const std = @import("std");

pub const AdamW = struct {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,

    pub fn step(
        self: *@This(),
        params: []f32,
        grads: []const f32,
        m: []f32,
        v: []f32,
        t: u32,
    ) void {
        for (params, 0..) |*param, i| {
            m[i] = self.beta1 * m[i] + (1 - self.beta1) * grads[i];
            v[i] = self.beta2 * v[i] + (1 - self.beta2) * grads[i] * grads[i];
            
            const m_hat = m[i] / (1 - std.math.pow(f32, self.beta1, t));
            const v_hat = v[i] / (1 - std.math.pow(f32, self.beta2, t));
            
            param.* -= self.learning_rate * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
        }
    }
};