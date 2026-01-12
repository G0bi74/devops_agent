# PostgreSQL Troubleshooting Guide

## Connection Refused

### Symptoms
- "connection refused" error
- Cannot connect to database

### Causes
1. PostgreSQL not running
2. Wrong port/host configuration
3. pg_hba.conf blocking connections
4. Firewall rules

### Solutions
```bash
# Check if running
systemctl status postgresql

# Check listening port
netstat -tlnp | grep 5432

# Check pg_hba.conf
cat /etc/postgresql/*/main/pg_hba.conf
```

## Too Many Connections

### Symptoms
- "FATAL: too many connections" error

### Solutions
1. Increase max_connections in postgresql.conf
2. Use connection pooling (PgBouncer)
3. Close idle connections
4. Check for connection leaks

## Slow Queries

### Diagnosis
1. Enable slow query log
2. Use EXPLAIN ANALYZE
3. Check for missing indexes

### Solutions
- Add appropriate indexes
- Optimize queries
- Increase shared_buffers
- Run VACUUM ANALYZE
