-- Candidate datasets for the build_cache_v38 RBV safety envelope.
-- ACX can verify that the required records exist, but the numeric test must
-- be applied per iteration after opening replica_path:
--   PNG:PP:VoltageProgram_RBV <= 5.0 V
--   PNG:QT3:CurrentProgram_RBV <= 10.0 A
--   PNG:QT9:CurrentProgram_RBV <= 10.0 A
--
-- Run:
-- sqlite3 /global/homes/b/bhowmic/campaign-store/IFE/laser.acx \
--   < queries/model_safety_envelope_candidates.sql

.headers on
.mode column

WITH
run_mode AS (
    SELECT archiveid, datasetid, trim(value, '"') AS mode
    FROM attributes
    WHERE name = '/run:mode'
),
event_count AS (
    SELECT archiveid, datasetid,
           CAST(trim(value, '"') AS INTEGER) AS recorded_events
    FROM attributes
    WHERE name = '/run:recordedEventCount'
),
pv_presence AS (
    SELECT archiveid, datasetid, trim(value, '"') AS pv
    FROM attributes
    WHERE name LIKE '/data/meshes/%/pv:name'
),
required_pv(pv) AS (
    VALUES
        ('PNG:PP:VoltageProgram_RBV'),
        ('PNG:QT3:CurrentProgram_RBV'),
        ('PNG:QT9:CurrentProgram_RBV')
)
SELECT
    ar.name AS archive,
    d.name AS dataset,
    COALESCE(ec.recorded_events, 0) AS recorded_events,
    d.replica_path,
    5.0 AS max_voltage_rbv_v,
    10.0 AS max_qt3_rbv_a,
    10.0 AS max_qt9_rbv_a,
    'candidate; enforce limits per iteration in BP5' AS result
FROM datasets AS d
JOIN archives AS ar ON ar.archiveid = d.archiveid
JOIN run_mode AS rm ON rm.archiveid = d.archiveid AND rm.datasetid = d.datasetid
LEFT JOIN event_count AS ec ON ec.archiveid = d.archiveid AND ec.datasetid = d.datasetid
WHERE rm.mode = 'shot'
  AND NOT EXISTS (
      SELECT 1 FROM required_pv AS r
      WHERE NOT EXISTS (
          SELECT 1 FROM pv_presence AS p
          WHERE p.archiveid = d.archiveid
            AND p.datasetid = d.datasetid
            AND p.pv = r.pv
      )
  )
ORDER BY ar.name, d.name;
