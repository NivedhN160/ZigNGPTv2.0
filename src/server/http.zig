const std = @import("std");
const net = std.net;

pub fn startServer(allocator: std.mem.Allocator, port: u16) !void {
    const address = try net.Address.parseIp4("0.0.0.0", port);
    var server = net.StreamServer.init(.{});
    try server.listen(address);

    std.debug.print("Server running on http://localhost:{}\n", .{port});

    while (true) {
        const conn = try server.accept();
        defer conn.stream.close();

        var buf: [1024]u8 = undefined;
        const len = try conn.stream.read(&buf);
        const request = buf[0..len];

        if (std.mem.indexOf(u8, request, "GET /chat") != null) {
            const response = 
                "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: text/plain\r\n" ++
                "\r\n" ++
                "Hello from ZigGPT!";
            _ = try conn.stream.write(response);
        }
    }
}