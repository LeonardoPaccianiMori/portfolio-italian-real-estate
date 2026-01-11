/**
 * Italian Real Estate Pipeline - MongoDB Initialization Script
 *
 * This script runs automatically when the MongoDB container first starts.
 * It creates the necessary databases and collections for the pipeline.
 *
 * Databases:
 *   - listing_website_datalake: Stores raw HTML data from web scraping
 *   - listing_website_warehouse: Stores transformed feature data after ETL
 *
 * Collections (same for both databases):
 *   - sale: Sale listings
 *   - rent: Rental listings
 *   - auction: Auction listings
 *
 * Author: Leonardo Pacciani-Mori
 * License: MIT
 */

print('============================================');
print('  MongoDB Initialization');
print('============================================');
print('');

// =============================================================================
// Create Datalake Database
// =============================================================================
print('Creating datalake database and collections...');

db = db.getSiblingDB('listing_website_datalake');

// Create collections with validation
db.createCollection('sale');
db.createCollection('rent');
db.createCollection('auction');

// Create indexes for common queries
db.sale.createIndex({ "listing_id": 1 }, { unique: false });
db.sale.createIndex({ "province": 1 });
db.sale.createIndex({ "scraped_at": 1 });

db.rent.createIndex({ "listing_id": 1 }, { unique: false });
db.rent.createIndex({ "province": 1 });
db.rent.createIndex({ "scraped_at": 1 });

db.auction.createIndex({ "listing_id": 1 }, { unique: false });
db.auction.createIndex({ "province": 1 });
db.auction.createIndex({ "scraped_at": 1 });

print('  - listing_website_datalake: OK');

// =============================================================================
// Create Warehouse Database
// =============================================================================
print('Creating warehouse database and collections...');

db = db.getSiblingDB('listing_website_warehouse');

// Create collections
db.createCollection('sale');
db.createCollection('rent');
db.createCollection('auction');

// Create indexes for ETL and migration queries
db.sale.createIndex({ "listing_id": 1 }, { unique: false });
db.sale.createIndex({ "province": 1 });
db.sale.createIndex({ "processed_at": 1 });

db.rent.createIndex({ "listing_id": 1 }, { unique: false });
db.rent.createIndex({ "province": 1 });
db.rent.createIndex({ "processed_at": 1 });

db.auction.createIndex({ "listing_id": 1 }, { unique: false });
db.auction.createIndex({ "province": 1 });
db.auction.createIndex({ "processed_at": 1 });

print('  - listing_website_warehouse: OK');

// =============================================================================
// Summary
// =============================================================================
print('');
print('============================================');
print('  MongoDB Initialization Complete');
print('============================================');
print('');
print('Created databases:');
print('  - listing_website_datalake (sale, rent, auction)');
print('  - listing_website_warehouse (sale, rent, auction)');
print('');
