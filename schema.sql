-- schema.sql

-- Drop tables if they exist (good for development, but remove in production)
DROP TABLE IF EXISTS users;

-- Create the users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT DEFAULT 'user'
);

--  No need to create tables for models.  Those are for the python scripts, not the database.
