# General DevOps Best Practices

## Service Health Checks

Always implement health check endpoints:
- `/health` - basic liveness
- `/ready` - readiness for traffic

## Log Management

### Log Levels
- ERROR: Application errors requiring attention
- WARNING: Potential issues
- INFO: Normal operations
- DEBUG: Detailed debugging info

### Log Rotation
Configure logrotate to prevent disk filling:
```
/var/log/app/*.log {
    daily
    rotate 7
    compress
    missingok
}
```

## Monitoring Essentials

Key metrics to monitor:
1. CPU usage
2. Memory usage
3. Disk space
4. Network I/O
5. Application response time
6. Error rates

## Incident Response

1. **Detect**: Automated alerts
2. **Respond**: On-call engineer notified
3. **Mitigate**: Reduce impact
4. **Resolve**: Fix root cause
5. **Learn**: Post-mortem
