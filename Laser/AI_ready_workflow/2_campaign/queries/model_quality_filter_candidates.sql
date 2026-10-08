-- Candidate datasets for the build_cache_v38 per-shot quality filter.
-- This query selects shot datasets containing every source record
-- needed to evaluate these rules after opening the BP5 replica:
--   PNG:ModeTab:Index == FULL_SYS_MODE
--   PNG:PP:Energy > 0.1 J
--   peak(Siglent:Ch4:Trace) >= 0.10 V
--   Siglent Ch1/Ch2/Ch4 Delay >= 0
--   0 < Siglent:Ch4:FWHM <= 50
--   Siglent:Ch4:Delay < 400
--   reject BNC default/off state:
--     Ch2 Delay < 0.2501 AND Ch3 Delay == 0 AND Ch4 Delay == 0
--
-- Run:
-- sqlite3 /global/homes/b/bhowmic/campaign-store/IFE/laser.acx \
--   < queries/model_quality_filter_candidates.sql

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
        ('PNG:ModeTab:Index'),
        ('PNG:PP:Energy'),
        ('Siglent:Ch4:Trace'),
        ('Siglent:Ch1:Delay'),
        ('Siglent:Ch2:Delay'),
        ('Siglent:Ch4:Delay'),
        ('Siglent:Ch4:FWHM'),
        ('BNC1:Ch2:Delay'),
        ('BNC1:Ch3:Delay'),
        ('BNC1:Ch4:Delay')
)
SELECT
    ar.name AS archive,
    d.name AS dataset,
    COALESCE(ec.recorded_events, 0) AS recorded_events,
    d.replica_path,
    'candidate; apply v38 rules per iteration in BP5' AS result
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
