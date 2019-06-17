import sqlite3
from sqlite3 import Error


class Database(object):
    def __init__(self, db_file):
        try:
            self.db = sqlite3.connect(db_file)
        except Error as e:
            logger.error('Could not load database file: %s' % db_file)

        self.cursor = self.db.cursor()

    def execute(self, query):
        try:
            self.cursor.execute(query)
            self.db.commit()
        except Error as e:
            self.db.rollback()
            raise e

        return self.cursor.fetchall()

    def create_tables(self, table_dict):
        for table_name, col_name_list in table_dict.items():
            self.execute("CREATE TABLE {} ({})".format(
                            table_name, ','.join(col_name_list)))

    def insert_row(self, table_name, val_list):
        return self.execute("INSERT INTO {} VALUES (\"{}\")".format(
                            table_name, '","'.join(val_list)))

    @property
    def tables(self):
        tables = self.execute("""
        SELECT name FROM sqlite_master
        WHERE name!='sqlite_sequence' and type='table'
        """)
        return [table[0] for table in tables]

    @property
    def columns(self):
        columns = set()
        for table in self.tables:
            for column in self.execute('pragma table_info({})'.format(table)):
                columns.add(column[1])
        return list(columns)

    def select_from_all(self, columns, constraint):
        columns = [col for col in columns if col in self.columns]
        columns_sql = ','.join(columns)
        tables_sql = ' NATURAL JOIN '.join(self.tables)
        constraint_sql = ['{}={}'.format(key, value)
                          for key, value in constraint.items()]
        sql = 'SELECT {} FROM {}'.format(columns_sql, tables_sql)
        if len(constraint) > 0:
            sql += ' WHERE {}'.format(' and '.join(constraint_sql))
        ret = zip(*self.execute(sql))
        return {key: values for key, values in zip(columns, ret)}

    def join_and_create_tables(self, table_name_list):
        table_name_comb = "comb"
        num_tables = len(table_name_list)
        if num_tables == 1:
            return table_name_list[0]
        else:
            query = "CREATE TABLE {} AS SELECT * FROM {} INNER JOIN {}".\
                    format(table_name_comb, table_name_list[0],
                           table_name_list[1])

            for i in range(2, num_tables):
                query += " INNER JOIN {}".format(table_name_list[i])

            self.execute(query)
        return table_name_comb
