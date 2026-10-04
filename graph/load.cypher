// Script nạp đồ thị bất động sản (Notebook 3). src/graph.py chạy từng khối theo tên, truyền $rows theo lô.
// Schema: (Listing)-[:OF_TYPE]->(PropertyType)
//         (Listing)-[:IN_WARD]->(Ward)-[:IN_DISTRICT]->(District)-[:IN_PROVINCE]->(Province)
// Tin thiếu phường nối thẳng vào quận (IN_DISTRICT); thiếu cả quận thì nối vào tỉnh (IN_PROVINCE).
// Khoá của Ward/District gồm cả cấp cha để hai địa danh trùng tên ở hai nơi không bị gộp.

// name: constraints
CREATE CONSTRAINT listing_id IF NOT EXISTS FOR (n:Listing) REQUIRE n.listing_id IS UNIQUE;
CREATE CONSTRAINT province_key IF NOT EXISTS FOR (n:Province) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT district_key IF NOT EXISTS FOR (n:District) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT ward_key IF NOT EXISTS FOR (n:Ward) REQUIRE n.key IS UNIQUE;
CREATE CONSTRAINT property_type_name IF NOT EXISTS FOR (n:PropertyType) REQUIRE n.name IS UNIQUE;
CREATE INDEX district_name IF NOT EXISTS FOR (n:District) ON (n.name);
CREATE INDEX ward_name IF NOT EXISTS FOR (n:Ward) ON (n.name);
CREATE INDEX listing_price IF NOT EXISTS FOR (n:Listing) ON (n.price);

// name: provinces
UNWIND $rows AS r
MERGE (p:Province {key: r.key})
SET p.name = r.name;

// name: districts
UNWIND $rows AS r
MERGE (d:District {key: r.key})
SET d.name = r.name
WITH d, r
MATCH (p:Province {key: r.parent})
MERGE (d)-[:IN_PROVINCE]->(p);

// name: wards
UNWIND $rows AS r
MERGE (w:Ward {key: r.key})
SET w.name = r.name
WITH w, r
MATCH (d:District {key: r.parent})
MERGE (w)-[:IN_DISTRICT]->(d);

// name: property_types
UNWIND $rows AS r
MERGE (:PropertyType {name: r.name});

// name: listings
UNWIND $rows AS r
MERGE (l:Listing {listing_id: r.listing_id})
SET l += r.props;

// name: of_type
UNWIND $rows AS r
MATCH (l:Listing {listing_id: r.listing_id})
MATCH (t:PropertyType {name: r.target})
MERGE (l)-[:OF_TYPE]->(t);

// name: in_ward
UNWIND $rows AS r
MATCH (l:Listing {listing_id: r.listing_id})
MATCH (w:Ward {key: r.target})
MERGE (l)-[:IN_WARD]->(w);

// name: in_district
UNWIND $rows AS r
MATCH (l:Listing {listing_id: r.listing_id})
MATCH (d:District {key: r.target})
MERGE (l)-[:IN_DISTRICT]->(d);

// name: in_province
UNWIND $rows AS r
MATCH (l:Listing {listing_id: r.listing_id})
MATCH (p:Province {key: r.target})
MERGE (l)-[:IN_PROVINCE]->(p);
