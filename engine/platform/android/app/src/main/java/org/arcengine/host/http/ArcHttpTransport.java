package org.arcengine.host.http;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;
import java.util.List;
import java.util.Map;

/**
 * Native ARC's bounded C9 HTTP client uses the Android platform's URLConnection
 * implementation for TLS trust, hostname verification and proxy policy.
 *
 * No arbitrary editor paths, credentials, certificate overrides or redirects
 * are accepted. C++ validates the immutable object/release URL before calling.
 * Response bodies are streamed in JNI-sized chunks, never buffered on Java heap.
 */
public final class ArcHttpTransport {
    private ArcHttpTransport() {}

    public static final class Response {
        public final int status;
        public final String contentLength;
        public final String contentRange;
        public final String etag;
        public final String redirectLocation;
        private final HttpURLConnection connection;
        private final InputStream body;

        Response(HttpURLConnection connection, int status, String length, String range,
                 String etag, String redirect, InputStream body) {
            this.connection = connection;
            this.status = status;
            this.contentLength = length;
            this.contentRange = range;
            this.etag = etag;
            this.redirectLocation = redirect;
            this.body = body;
        }

        public int read(byte[] buffer) throws IOException {
            return body == null ? -1 : body.read(buffer);
        }

        public void close() {
            if (body != null) {
                try {
                    body.close();
                } catch (IOException ignored) {
                    // Disconnect is mandatory even after a read/close failure.
                }
            }
            connection.disconnect();
        }
    }

    private static boolean loopback(String host) {
        return "localhost".equalsIgnoreCase(host) || "127.0.0.1".equals(host)
                || "::1".equals(host) || "[::1]".equals(host);
    }

    private static String singleHeader(HttpURLConnection connection, String name) throws IOException {
        String value = null;
        for (Map.Entry<String, List<String>> entry : connection.getHeaderFields().entrySet()) {
            if (entry.getKey() == null || !entry.getKey().equalsIgnoreCase(name)) continue;
            List<String> values = entry.getValue();
            if (value != null || values == null || values.size() != 1 || values.get(0) == null
                    || values.get(0).length() > 8192) {
                throw new IOException("Invalid duplicate or oversized HTTP response header");
            }
            value = values.get(0);
        }
        return value == null ? "" : value;
    }

    /**
     * @param address fully validated absolute object or release URL
     * @param range "bytes=start-end" or empty
     * @param ifMatch strong ETag or empty
     * @param noCache true for mutable current.json release descriptors
     * @param timeoutMs bounded connect/read timeout; >0
     */
    public static Response open(String address, String range, String ifMatch,
                                boolean noCache, int timeoutMs) throws IOException {
        if (timeoutMs <= 0) throw new IOException("HTTP timeout must be positive");
        URL url = new URL(address);
        boolean local = loopback(url.getHost());
        if (!"https".equalsIgnoreCase(url.getProtocol()) &&
                !("http".equalsIgnoreCase(url.getProtocol()) && local)) {
            throw new IOException("Android HTTP requires HTTPS or explicit loopback");
        }
        // The caller already enforced the development-only loopback flag. Keep
        // the platform helper equally restrictive if used by other native code.
        HttpURLConnection connection = (HttpURLConnection) (local
                ? url.openConnection(Proxy.NO_PROXY) : url.openConnection());
        try {
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(timeoutMs);
            connection.setReadTimeout(timeoutMs);
            connection.setUseCaches(false);
            connection.setAllowUserInteraction(false);
            connection.setRequestMethod("GET");
            connection.setRequestProperty("Accept-Encoding", "identity");
            if (!range.isEmpty()) connection.setRequestProperty("Range", range);
            if (!ifMatch.isEmpty()) connection.setRequestProperty("If-Match", ifMatch);
            if (noCache) {
                connection.setRequestProperty("Cache-Control", "no-cache, no-store");
                connection.setRequestProperty("Pragma", "no-cache");
            }

            int status = connection.getResponseCode();
            String length = singleHeader(connection, "Content-Length");
            String contentRange = singleHeader(connection, "Content-Range");
            String etag = singleHeader(connection, "ETag");
            String location = singleHeader(connection, "Location");
            String encoding = singleHeader(connection, "Content-Encoding");
            if (!encoding.isEmpty() && !"identity".equalsIgnoreCase(encoding)) {
                throw new IOException("Unexpected encoded HTTP body");
            }
            InputStream body = (status == HttpURLConnection.HTTP_OK ||
                    status == HttpURLConnection.HTTP_PARTIAL) ? connection.getInputStream() : null;
            return new Response(connection, status, length, contentRange, etag, location, body);
        } catch (IOException | RuntimeException error) {
            connection.disconnect();
            throw error;
        }
    }
}
