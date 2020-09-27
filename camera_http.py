from http import server
import cv2
import socketserver
import json
import cgi

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/labels':
            ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
            if ctype != 'application/json':
                self.send_response(400)
                self.end_headers()
                return
            length = int(self.headers["Content-Length"])
            raw_message = self.rfile.read(length)
            message = json.loads(raw_message)
            print(message['label'])
            print(message['pos'])
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'status': 0}), 'utf-8'))

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    cam = cv2.VideoCapture(0)
                    ret, frame = cam.read()
                    result, frame = cv2.imencode('.jpg', frame, encode_param)
                    frame = cv2.flip(frame, 1)
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                print(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def run(server_class=server.HTTPServer, handler_class=StreamingHandler, port=8008):
    server_address = ('', port)
    httpd = StreamingServer(server_address, handler_class)

    print('Starting httpd on port {}...'.format(port))
    httpd.serve_forever()

run()
