-- View chuẩn trên bảng sạch (data-contract §3.1). vre/data/duck.py thay {clean_path} bằng settings.clean_path tuyệt đối.
CREATE OR REPLACE VIEW v_listings AS
SELECT *,
  price_vnd / 1e9                                              AS price_bn,
  CASE WHEN NOT is_price_outlier AND NOT is_area_outlier AND price_vnd IS NOT NULL AND area_m2 IS NOT NULL
       THEN price_vnd / area_m2 / 1e6 END                      AS price_m2_mil,
  strftime(published_at, '%Y-%m')                              AS month
FROM read_parquet('{clean_path}');
