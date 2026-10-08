-- Count indexed Laser datasets whose run mode is shot or alignment.
.headers on
.mode column

SELECT
    trim(value, '"') AS run_mode,
    COUNT(*) AS datasets
FROM attributes
WHERE name = '/run:mode'
  AND trim(value, '"') IN ('shot', 'alignment')
GROUP BY run_mode
ORDER BY datasets DESC;
