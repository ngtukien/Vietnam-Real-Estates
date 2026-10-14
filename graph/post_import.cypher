// Constraint + index sau `neo4j-admin database import full` (data-contract §6.3).
// vre.index.graph_import chạy lần lượt từng lệnh (tách theo dấu `;`, bỏ dòng `//`), rồi đợi mọi index ONLINE.
// Mọi lệnh có IF NOT EXISTS nên chạy lại an toàn.

// name: constraints
CREATE CONSTRAINT listing_id IF NOT EXISTS FOR (n:Listing) REQUIRE n.listing_id IS UNIQUE;
CREATE CONSTRAINT property_type_name IF NOT EXISTS FOR (n:PropertyType) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT province_key IF NOT EXISTS FOR (n:Province) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT district_key IF NOT EXISTS FOR (n:District) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT ward_key IF NOT EXISTS FOR (n:Ward) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT street_key IF NOT EXISTS FOR (n:Street) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT project_key IF NOT EXISTS FOR (n:Project) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT month_key IF NOT EXISTS FOR (n:Month) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT amenity_key IF NOT EXISTS FOR (n:Amenity) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (n:Cluster) REQUIRE n.cluster_id IS UNIQUE;
CREATE CONSTRAINT market_stat_key IF NOT EXISTS FOR (n:MarketStat) REQUIRE n.key IS UNIQUE;

// name: range_indexes
CREATE INDEX district_name IF NOT EXISTS FOR (n:District) ON (n.name);
CREATE INDEX ward_name IF NOT EXISTS FOR (n:Ward) ON (n.name);
CREATE INDEX street_name IF NOT EXISTS FOR (n:Street) ON (n.name);
CREATE INDEX project_name IF NOT EXISTS FOR (n:Project) ON (n.name);
CREATE INDEX listing_price_vnd IF NOT EXISTS FOR (n:Listing) ON (n.price_vnd);
CREATE INDEX listing_area_m2 IF NOT EXISTS FOR (n:Listing) ON (n.area_m2);
CREATE INDEX listing_price_m2_mil IF NOT EXISTS FOR (n:Listing) ON (n.price_m2_mil);
CREATE INDEX listing_bedrooms IF NOT EXISTS FOR (n:Listing) ON (n.bedrooms);
CREATE INDEX listing_month IF NOT EXISTS FOR (n:Listing) ON (n.month);
CREATE INDEX market_stat_lookup IF NOT EXISTS FOR (n:MarketStat) ON (n.scope, n.property_type, n.month, n.deal_type);

// name: fulltext_indexes
// CHỈ code API/resolver dùng; Cypher do LLM sinh bị cấm `CALL` (vre.graph.guard).
CREATE FULLTEXT INDEX project_name_ft IF NOT EXISTS FOR (n:Project) ON EACH [n.name];
CREATE FULLTEXT INDEX street_name_ft IF NOT EXISTS FOR (n:Street) ON EACH [n.name];
