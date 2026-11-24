import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.arcengine.host.http.ArcHttpTransport;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Java SE loopback contract tests for Android's platform HTTP adapter.
 * Run without an emulator; the Android APK CI also compiles the same helper.
 */
public final class ArcHttpTransportHostTest {
    private static void check(boolean condition, String reason) {
        if (!condition) throw new AssertionError(reason);
    }

    private static String text(ArcHttpTransport.Response response) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] chunk = new byte[3]; // Prove chunked Java/JNI read contract.
        for (int length; (length = response.read(chunk)) >= 0;) {
            if (length > 0) output.write(chunk, 0, length);
        }
        return output.toString(StandardCharsets.UTF_8);
    }

    private static void reply(HttpExchange request, int status, String body) throws IOException {
        byte[] data = body.getBytes(StandardCharsets.UTF_8);
        request.sendResponseHeaders(status, data.length);
        if (data.length > 0) request.getResponseBody().write(data);
        request.close();
    }

    public static void main(String[] unused) throws Exception {
        HttpServer server = HttpServer.create(
                new InetSocketAddress(InetAddress.getByName("127.0.0.1"), 0), 0);
        AtomicReference<String> range = new AtomicReference<>();
        AtomicReference<String> ifMatch = new AtomicReference<>();
        AtomicReference<String> noCache = new AtomicReference<>();
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicReference<String> objectAuthorization = new AtomicReference<>();
        server.createContext("/objects", request -> {
            range.set(request.getRequestHeaders().getFirst("Range"));
            ifMatch.set(request.getRequestHeaders().getFirst("If-Match"));
            objectAuthorization.set(request.getRequestHeaders().getFirst("Authorization"));
            request.getResponseHeaders().set("ETag", "\"sha-immutable\"");
            if (range.get() == null) {
                reply(request, 200, "abcdefgh");
            } else {
                request.getResponseHeaders().set("Content-Range", "bytes 2-4/8");
                reply(request, 206, "cde");
            }
        });
        server.createContext("/releases", request -> {
            noCache.set(request.getRequestHeaders().getFirst("Cache-Control"));
            authorization.set(request.getRequestHeaders().getFirst("Authorization"));
            reply(request, 200, "{\"generation\":6}");
        });
        server.createContext("/redirect", request -> {
            request.getResponseHeaders().set("Location", "https://evil.invalid/object");
            reply(request, 302, "");
        });
        server.createContext("/encoded", request -> {
            request.getResponseHeaders().set("Content-Encoding", "gzip");
            reply(request, 200, "hello");
        });
        server.start();
        try {
            String origin = "http://127.0.0.1:" + server.getAddress().getPort();
            ArcHttpTransport.Response full = ArcHttpTransport.open(
                    origin + "/objects", "", "", false, 5000);
            try {
                check(full.status == 200, "full-object status");
                check("\"sha-immutable\"".equals(full.etag), "immutable object ETag");
                check("abcdefgh".equals(text(full)), "full-object body");
            } finally {
                full.close();
            }

            ArcHttpTransport.Response partial = ArcHttpTransport.open(
                    origin + "/objects", "bytes=2-4", "\"sha-immutable\"", false, 5000);
            try {
                check(partial.status == 206, "range status");
                check("bytes 2-4/8".equals(partial.contentRange), "Content-Range");
                check("cde".equals(text(partial)), "ranged object bytes");
                check("bytes=2-4".equals(range.get()), "Range request header");
                check("\"sha-immutable\"".equals(ifMatch.get()), "If-Match request header");
            } finally {
                partial.close();
            }

            ArcHttpTransport.Response release = ArcHttpTransport.open(
                    origin + "/releases", "", "", true, 5000);
            try {
                check(release.status == 200, "release status");
                check("{\"generation\":6}".equals(text(release)), "release bytes");
                check("no-cache, no-store".equals(noCache.get()), "release no-cache header");
            } finally {
                release.close();
            }

            // An anonymous request must not synthesize pairing authority.
            check(authorization.get() == null, "anonymous request unexpectedly authenticated");
            check(objectAuthorization.get() == null, "anonymous object unexpectedly authenticated");
            String pairedToken = "b".repeat(64);
            ArcHttpTransport.Response pairedRange = ArcHttpTransport.open(
                    origin + "/objects", "bytes=2-4", "\"sha-immutable\"", pairedToken, false, 5000);
            try {
                check(pairedRange.status == 206, "paired range status");
                check("cde".equals(text(pairedRange)), "paired range body");
                check(("Bearer " + pairedToken).equals(objectAuthorization.get()),
                        "paired immutable object Authorization header");
            } finally {
                pairedRange.close();
            }
            ArcHttpTransport.Response paired = ArcHttpTransport.open(
                    origin + "/releases", "", "", pairedToken, true, 5000);
            try {
                check(paired.status == 200, "paired release status");
                check(("Bearer " + pairedToken).equals(authorization.get()), "paired Authorization header");
                check("{\"generation\":6}".equals(text(paired)), "paired release body");
            } finally {
                paired.close();
            }
            boolean invalidBearerRejected = false;
            try {
                ArcHttpTransport.open(origin + "/releases", "", "",
                        "invalid\r\nInjected-Header", true, 5000);
            } catch (IOException expected) {
                invalidBearerRejected = true;
            }
            check(invalidBearerRejected, "unsafe bearer token must fail before network request");

            boolean offHostBearerRejected = false;
            try {
                ArcHttpTransport.open("https://example.org/releases", "", "",
                        pairedToken, true, 5000);
            } catch (IOException expected) {
                offHostBearerRejected = true;
            }
            check(offHostBearerRejected, "paired credential must not leave the trusted loopback host");

            ArcHttpTransport.Response redirect = ArcHttpTransport.open(
                    origin + "/redirect", "", "", false, 5000);
            try {
                check(redirect.status == 302, "must not follow redirects");
                check("https://evil.invalid/object".equals(redirect.redirectLocation),
                        "untrusted Location must be visible for native validator");
            } finally {
                redirect.close();
            }

            boolean encodedRejected = false;
            try {
                ArcHttpTransport.open(origin + "/encoded", "", "", false, 5000);
            } catch (IOException expected) {
                encodedRejected = true;
            }
            check(encodedRejected, "unexpected content encoding must fail");

            boolean lanRejected = false;
            try {
                ArcHttpTransport.open("http://192.0.2.1:8080/objects", "", "", false, 5000);
            } catch (IOException expected) {
                lanRejected = true;
            }
            check(lanRejected, "plaintext LAN must fail before connection");
            System.out.println("Android ARC HTTP Java loopback contract tests passed");
        } finally {
            server.stop(0);
        }
    }
}
