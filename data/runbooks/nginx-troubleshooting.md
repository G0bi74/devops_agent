# Nginx Troubleshooting Guide

## Error 502 Bad Gateway

### Symptoms
- Users see "502 Bad Gateway" error
- Upstream servers not responding

### Common Causes
1. **Backend server down**: The application server (PHP-FPM, Node.js, etc.) is not running
2. **Socket issues**: Unix socket or TCP connection refused
3. **Timeout**: Backend is too slow to respond
4. **Memory issues**: Backend crashed due to OOM

### Diagnosis Steps
1. Check nginx error logs: `tail -f /var/log/nginx/error.log`
2. Verify backend is running: `systemctl status php-fpm`
3. Check socket/port: `netstat -tlnp | grep 9000`

### Solutions
- Restart backend: `systemctl restart php-fpm`
- Increase proxy timeouts in nginx config
- Check backend memory limits

## Error 504 Gateway Timeout

### Symptoms
- Request takes too long and fails

### Solutions
- Increase `proxy_read_timeout` in nginx config
- Optimize backend queries
- Add caching layer
