# Docker Troubleshooting Guide

## Container Exit Code 137 (OOM Killed)

### Symptoms
- Container exits with code 137
- dmesg shows OOM killer invoked

### Causes
- Container memory limit too low
- Memory leak in application
- Peak memory usage exceeded limit

### Solutions
1. Increase memory limit: `docker run -m 2g ...`
2. Add swap: `docker run --memory-swap 4g ...`
3. Find memory leaks in application
4. Use memory profiling tools

## Container Won't Start

### Common Issues
1. **Port conflict**: Port already in use
2. **Volume mount failed**: Path doesn't exist
3. **Image not found**: Pull the image first

### Diagnosis
```bash
docker logs <container_id>
docker inspect <container_id>
docker events
```

## Disk Space Issues

### Symptoms
- "no space left on device" errors
- Container creation fails

### Solutions
```bash
# Remove unused images
docker image prune -a

# Remove stopped containers
docker container prune

# Clean build cache
docker builder prune
```
