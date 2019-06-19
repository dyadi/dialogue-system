import sqlite3
create_statements = {}
create_statements['create_movie'] = """
CREATE TABLE IF NOT EXISTS Movie(
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    moviename NOT NULL,
    movie_series,
    actor,
    actress
);
"""
create_statements['create_theater'] = """
CREATE TABLE Theater (
    theater_id INTEGER PRIMARY KEY AUTOINCREMENT,
    theater NOT NULL,
    city
);
"""
create_statements['create_time_table'] = """
CREATE TABLE TimeTable (
    movie_id INTEGER,
    theater_id INTEGER,
    date,
    starttime,
    FOREIGN KEY(movie_id) REFERENCES Movie(movie_id),
    FOREIGN KEY(theater_id) REFERENCES Theater(theater_id)
);
"""

insert_movie = 'INSERT INTO MOVIE (moviename, actor) VALUES(?, ?)'
insert_theater = 'INSERT INTO Theater (theater, city) VALUES(?, ?)'
insert_time_table = 'INSERT INTO TimeTable (movie_id, theater_id, date, starttime) VALUES(?, ?, ?, ?)'

db = sqlite3.connect('movie.db')
cursor = db.cursor()
try:
    for key, stmt in create_statements.items():
        print(stmt)
        cursor.execute(stmt)
        db.commit()

    cursor.execute(insert_movie, ('Introduction to Computer Programming', 'Pang-Feng Liu'))
    cursor.execute(insert_theater, ('CSIE 204', 'Taipei'))
    cursor.execute(insert_time_table, (1, 1, '7/1', '9:00'))
    db.commit()
except sqlite3.Error as e:
    db.rollback()
    raise e
