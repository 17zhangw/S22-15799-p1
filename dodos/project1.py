from plumbum import cmd
from plumbum.cmd import grep, awk

from dodos import VERBOSITY_DEFAULT

DEFAULT_DB = "project1db"
DEFAULT_USER = "project1user"
DEFAULT_PASS = "project1pass"

# Note that pgreplay requires the following configuration:
#
# log_min_messages=error (or more)
# log_min_error_statement=log (or more)
# log_connections=on
# log_disconnections=on
# log_line_prefix='%m|%u|%d|%c|' (if you don't use CSV logging)
# log_statement='all'
# lc_messages must be set to English (encoding does not matter)
# bytea_output=escape (from version 9.0 on, only if you want to replay the log on 8.4 or earlier)
#
# Additionally, doit has a bit of an anti-feature with command substitution,
# so you have to escape %'s by Python %-formatting rules (no way to disable this behavior).

def task_project1_setup():
    mem_limit = int(awk['/MemAvailable/ { printf \"%d\", $2/1024/1024 }', "/proc/meminfo"]())
    num_cpu = int(grep["-c", "^processor", "/proc/cpuinfo"]())
    echo_list = [
        "max_connections=100",
        f"shared_buffers={int(mem_limit/4)}GB",
        f"effective_cache_size={int(mem_limit*3/4)}GB",
        f"maintenance_work_mem={int(mem_limit/16)}GB",
        "min_wal_size=1GB",
        "max_wal_size=2GB",
        "checkpoint_completion_target=0.9",
        "wal_buffers=16MB",
        "default_statistics_target=100",
        "random_page_cost=1.1",
        "effective_io_concurrency=100",
        f"max_parallel_workers={num_cpu}",
        f"max_worker_processes={num_cpu}",
        f"max_parallel_workers_per_gather=4",
        f"max_parallel_maintenance_workers=4",
        f"work_mem={int(1024*1024*(mem_limit*3/4)/(300)/4)}kB",
        "shared_preload_libraries='pg_stat_statements,pg_qualstats'",
        "pg_qualstats.track_constants=OFF",
        "pg_qualstats.sample_rate=1",
        "pg_qualstats.resolve_oids=True",
        "compute_query_id=ON",
    ]
    echos = "\n".join(echo_list)

    return {
        "actions": [
            # Install dependencies
            lambda: cmd.sudo["apt-get"]["install", "postgresql-14-hypopg"].run_fg(),
            lambda: cmd.sudo["apt-get"]["install", "postgresql-14-pg-qualstats"].run_fg(),
            "git submodule update --init --recursive",
            "pip install pandas",
            "pip install psycopg2",
            "pip install pglast",
            "cd behavior/modeling/featurewiz",
            "pip install -r requirements.txt",
            "cd ../../../",
            "pip install -r requirements.txt",

            # Open the models
            "mkdir -p artifacts",
            "cp models.tgz artifacts/",
            "cd artifacts/",
            "tar zxf models.tgz",
            "cd ..",

            "rm -rf blocklist.txt",
            "rm -rf pending.txt",
            
            lambda: cmd.sudo["bash"]["-c", f"echo \"{echos}\" >> /etc/postgresql/14/main/postgresql.conf"].run_fg(),
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_project1():

    def construct_index_name(base_tbl, index_cols, include_cols):
        return "idx_" + base_tbl + "_keys_" + ("_".join(index_cols)) + "_inc_" + ("_".join(include_cols))

    def reverse_index_sql(index_name):
        component0 = index_name.split("idx_")[1]
        component1 = component[0].split("_keys_")
        component2 = component1[1].split("_inc_")

        tbl_name = component1[0]
        index_cols = component2[0]
        include_cols = component2[1]
        if len(include_cols) == 0:
            return f"CREATE INDEX {index_name} ON {tbl_name}({index_cols})"
        else:
            return f"CREATE INDEX {index_name} ON {tbl_name}({index_cols}) INCLUDE ({include_cols})"

    def execute_query(connection, query, output_dict=False, key=None):
        from psycopg2.extras import RealDictCursor
        if output_dict:
            records = {}
        else:
            records = []

        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            for record in cursor:
                if output_dict:
                    records[record[key]] = record
                else:
                    records.append(record)

        return records

    def fetch_useless_indexes(connection):
        unused_query = """
            SELECT s.indexrelname AS indexname
              FROM pg_catalog.pg_stat_user_indexes s
              JOIN pg_catalog.pg_index i ON s.indexrelid = i.indexrelid
             WHERE s.idx_scan = 0      -- has never been scanned
               AND 0 <>ALL (i.indkey)  -- no index column is an expression
               AND NOT i.indisunique   -- is not a UNIQUE index
               AND NOT EXISTS          -- does not enforce a constraint
                   (SELECT 1 FROM pg_catalog.pg_constraint c WHERE c.conindid = s.indexrelid)
          ORDER BY pg_relation_size(s.indexrelid) DESC;
        """
        return [record['indexname'] for record in execute_query(connection, unused_query)]

    def compute_existing_indexes(connection):
        existing_indexes_sql_query = """
            SELECT statidx.relid,
                   statidx.indexrelid,
                   statidx.relname,
                   statidx.indexrelname,
                   idx.indnatts,
                   idx.indnkeyatts,
                   STRING_AGG(att.attname, ',') as columns
              FROM pg_stat_user_indexes statidx,
                   pg_index idx,
                   pg_attribute att
             WHERE idx.indexrelid = statidx.indexrelid
               AND att.attrelid = statidx.relid
               AND att.attnum = ANY(idx.indkey)
               AND array_position(idx.indkey, att.attnum) < idx.indnkeyatts
               AND att.atttypid != 0
               AND idx.indisunique = False
               AND idx.indisprimary = False
               AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint c WHERE c.conindid = statidx.indexrelid)
          GROUP BY statidx.relid,
                   statidx.indexrelid,
                   statidx.relname,
                   statidx.indexrelname,
                   idx.indisunique,
                   idx.indnatts,
                   idx.indnkeyatts
        """

        existing_indexes_include_sql_query = """
            SELECT statidx.relid,
                   statidx.indexrelid,
                   statidx.relname,
                   statidx.indexrelname,
                   idx.indnatts,
                   idx.indnkeyatts,
                   STRING_AGG(att.attname, ',') as columns
              FROM pg_stat_user_indexes statidx,
                   pg_index idx,
                   pg_attribute att
             WHERE idx.indexrelid = statidx.indexrelid
               AND att.attrelid = statidx.relid
               AND att.attnum = ANY(idx.indkey)
               AND array_position(idx.indkey, att.attnum) >= idx.indnkeyatts
               AND att.atttypid != 0
               AND idx.indisunique = False
               AND idx.indisprimary = False
               AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint c WHERE c.conindid = statidx.indexrelid)
          GROUP BY statidx.relid,
                   statidx.indexrelid,
                   statidx.relname,
                   statidx.indexrelname,
                   idx.indisunique,
                   idx.indnatts,
                   idx.indnkeyatts
        """

        existing_indexes = execute_query(connection, existing_indexes_sql_query)
        existing_indexes_include = execute_query(connection, existing_indexes_include_sql_query, True, 'indexrelname')
        indexes = {}
        for record in existing_indexes:
            index_name = record['indexrelname']
            include_columns = []
            if index_name in existing_indexes_include:
                include_columns = existing_indexes_include[index_name]['columns'].split(',')

            index_columns = record['columns'].split(',')
            if len(index_columns) != record['indnkeyatts']:
                # malformed index entry
                continue

            if len(index_columns) + len(include_columns) - len(set(index_columns).intersection(set(include_columns))) != record['indnatts']:
                # there might be a duplicate or it's malformed
                continue

            marked_name = construct_index_name(record["relname"], index_columns, include_columns)
            indexes[marked_name] = {
                'relname': record["relname"],
                'indexrelname': index_name,
                'index_columns': index_columns,
                'include_columns': include_columns
            }

        return indexes

    def get_indexadvisor(connection):
        sql_query = "SELECT v FROM json_array_elements(pg_qualstats_index_advisor(min_filter=>0, min_selectivity=>0)->'indexes') v ORDER BY v::text COLLATE \"C\";"
        indexes = {}
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            for record in cursor:
                components = record[0].split(' ON ')
                table = components[1].split(' ')[0]
                if '.' in table:
                    table = table.split('.')[1]

                fields_start = record[0].split('(')[1]
                fields_str = fields_start.split(')')[0]
                fields_str = fields_str.replace(' ', '')
                fields = fields_str.split(',')
                if len(fields) == 0:
                    continue

                indexrelname = construct_index_name(table, fields, [])
                indexes[indexrelname] = {
                    'relname': table,
                    'indexrelname': indexrelname,
                    'index_columns': fields,
                    'include_columns': []
                }

        return indexes

    def mutate(connection, existing_indexes):
        import random
        tables = {}

        sql_query = """
            SELECT tbl.relid, tbl.relname, STRING_AGG(att.attname, ',') as columns
            FROM pg_stat_user_tables tbl, pg_attribute att
            WHERE tbl.relid = att.attrelid AND NOT att.attisdropped AND att.attnum > 0
            GROUP BY tbl.relid, tbl.relname
        """
        results = execute_query(connection, sql_query)
        for record in results:
            tables[record['relname']] = set(record['columns'].split(','))

        candidates = {}
        for (index, value) in existing_indexes.items():
            rel = value['relname']
            if rel not in tables:
                continue

            index_columns = value['index_columns']
            include_columns = value['include_columns']
            valid_new_index_columns = tables[rel] - set(index_columns) - set(include_columns)
            if len(valid_new_index_columns) == 0:
                continue

            new_index = random.choice(tuple(valid_new_index_columns))
            new_include = random.choice(tuple(valid_new_index_columns))
            new_index_name = construct_index_name(rel, index_columns + [new_index], include_columns)
            new_include_name = construct_index_name(rel, index_columns, include_columns + [new_include])
            candidates[new_index_name] = {
                'relname': rel,
                'indexrelname': new_index_name,
                'index_columns': index_columns + [new_index],
                'include_columns': include_columns
            }

            candidates[new_include_name] = {
                'relname': rel,
                'indexrelname': new_include_name,
                'index_columns': index_columns,
                'include_columns': include_columns + [new_include]
            }

        for i in range(5):
            # This is the "gifted child phase of this process".
            table_set = [key for (key, _) in tables.items()]
            random_table = random.choice(tuple(table_set))
            random_column = random.choice(tuple(tables[random_table]))
            name = construct_index_name(random_table, [random_column], [])
            candidates[name] = {
                'relname': random_table,
                'indexrelname': name,
                'index_columns': [random_column],
                'include_columns': []
            }

        return candidates

    def eliminate_candidates(candidates, blocklist, pending):
        import random
        # Remove any candidates that have been blocked already.
        remove_list = []
        for candidate in candidates:
            if candidate in blocklist:
                remove_list.append(candidate)
            elif candidate in pending:
                remove_list.append(candidate)
        [candidates.pop(remove) for remove in remove_list]

        # Append candidates and shuffle the list
        candidate_keys = [key for (key, value) in candidates.items()]
        candidate_list = list(pending.union(candidate_keys))
        random.shuffle(candidate_list)

        # At-most only evaluate X many candidates with models.
        return candidate_list[0:5], candidate_list[5:]

    def evaluate(connection, workloads, candidates):
        import pickle
        from pathlib import Path
        from operator import itemgetter
        model_dict = {}
        for ou_type_path in Path("artifacts/gbm/").glob("*"):
            ou_type = ou_type_path.name
            model_path = list(ou_type_path.glob("*.pkl"))
            assert len(model_path) == 1
            model_path = model_path[0]
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
                model_dict[ou_type] = model

        workload_features = {}
        candidate_costs = {}
        with connection.cursor() as cursor:
            for candidate in candidates:
                candidate_cost = 0
                sql = reverse_index_sql(candidate)
                cursor.execute(f"SELECT * FROM hypopg_create_index('{sql}')")
                for workload in workloads:
                    cursor.execute(f"EXPLAIN (FORMAT JSON) {workload}")
                    for record in cursor:
                        json = record[0]

                    try:
                        def diff(plan):
                            if 'Plans' not in plan:
                                return

                            for plan_obj in plan['Plans']:
                                if plan_obj['Startup Cost'] >= plan['Startup Cost']:
                                    plan['Startup Cost'] = 0.00001
                                else:
                                    plan['Startup Cost'] -= plan_obj['Startup Cost']

                                if plan_obj['Total Cost'] >= plan['Total Cost']:
                                    plan['Total Cost'] = 0.00001
                                else:
                                    plan['Total Cost'] -= plan_obj['Total Cost']
                                diff(plan_obj)

                        def accumulate(plan):
                            if 'Plans' in plan:
                                for plan_obj in plan['Plans']:
                                    accumulate(plan_obj)

                            ou_type = plan['Node Type'].replace(' ', '')
                            rows = float(plan['Plan Rows']) if plan['Plan Rows'] > 0 else 0.00001
                            width = float(plan['Plan Width']) if plan['Plan Width'] > 0 else 0.00001
                            startup = plan['Startup Cost'] if plan['Startup Cost'] > 0 else 0.00001
                            total = plan['Total Cost'] if plan['Total Cost'] > 0 else 0.00001
                            x = [rows, width, startup, total]
                            candidate_cost = candidate_cost + model_dict[ou_type].predict(x)[0][-1]

                        root = json[0]['Plan']
                        diff(root)
                        accumulate(root)
                    except Exception as e:
                        pass

                cursor.execute(f"SELECT hypopg_reset()")
                candidate_costs[candidate] = candidate_cost

        result = dict(sorted(candidate_costs.items(), key=itemgetter(1))[:2])
        for k, v in result:
            candidate_costs.pop(k)
        return [k for (k, _) in result.items()], [k for (k, _) in candidate_costs.items()]


    def derive_actions(workload_csv, timeout):
        import psycopg2

        blocklist = set()
        try:
            with open("blocklist.txt", "r") as f:
                blocklist.extend(f.readlines())
        except:
            pass

        pending = set()
        try:
            with open("pending.txt", "r") as f:
                pending.extend(f.readlines())
        except:
            pass

        actions = []
        with psycopg2.connect("host=localhost dbname=project1db user=project1user password=project1pass") as connection:
            # Turn on auto-commit.
            connection.set_session(autocommit=True)
            workloads = process_workload(workload_csv)

            connection.cursor().execute("CREATE EXTENSION IF NOT EXISTS hypopg")

            # Get set of useless indexes.
            useless_indexes = fetch_useless_indexes(connection)
            # Compute the existing set of indexes.
            existing_indexes = compute_existing_indexes(connection)

            # Always drop any index that is in useless_indexes
            remove_list = []
            orig_names = {value['indexrelname']: key for (key, value) in existing_indexes.items()}
            for index in useless_indexes:
                if index in orig_names:
                    remove_list.append(orig_names[index])
                blocklist.add(index)
                actions.append(f"DROP INDEX IF EXISTS {index}")
            [existing_indexes.pop(index) for index in remove_list]

            # get_indexadvisor() definition
            advised_indexes = get_indexadvisor(connection)

            # let the mutations of the future decide the true course...
            candidates = mutate(connection, existing_indexes)

            # eliminate candidates based on pending and blocklist
            candidates, pending = eliminate_candidates(candidates, blocklist, pending)

            # evaluate hypothetical index performances
            candidates, unselected = evaluate(connection, workloads, candidates)
            pending.extend(unselected)
            actions.append("SELECT hypopg_reset()")
            for candidate in candidates:
                actions.append(reverse_index_sql(candidate))


        with open("blocklist.txt", "w") as f:
            for item in blocklist:
                f.write(item + "\n")

        with open("pending.txt", "w") as f:
            items = pending[0:20]
            for item in items:
                f.write(pending + "\n")

        actions.append("SELECT pg_qualstats_reset()")
        with open("actions.sql", "w") as f:
            for action in actions:
                f.write(action)
                f.write("\n")

        print("Done generating actions for round")

    def process_workload(workload_csv):
        import pandas as pd
        import pglast
        data = pd.read_csv(workload_csv, header=None)

        filters = [
            "statement: begin",
            "statement: alter system set",
            "statement: set",
            "statement: rollback",
            "statement: commit",
            "create extension",
            "hypopg",
            "pg_catalog",
            "pg_qualstats",
            "pg_stat",
            "current_schema",
            "version",
            "show",
        ]

        selection_vector = data[13].str.contains("statement: ")
        selection_vector = selection_vector & (~(data[13] == "statement: "))
        for kw in filters:
            selection_vector = selection_vector & (~data[13].str.lower().str.contains(kw))

        data = data.loc[selection_vector][13]
        data = data.str.replace("statement: ", "")
        data.reset_index(inplace=True, drop=True)
        fingerprint = []
        queries = []
        for query in data:
            queries.append(query)
            fingerprint.append(pglast.parser.fingerprint(query))

        data = pd.DataFrame.from_records(zip(list(data), fingerprint), columns=["statement", "fingerprint"])
        def apply_func(df):
            row = df.head(1)
            row["frequency"] = df.shape[0]
            return row
        data = data.groupby("fingerprint").apply(apply_func)
        data.sort_values(by=["frequency"], ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)

        output = {}
        for rec in data.itertuples():
            output[rec[1]] = rec[3]
        return output

    return {
        "actions": [derive_actions],
        "uptodate": [False],
        "verbosity": 2,
        "params": [
            {
                "name": "workload_csv",
                "long": "workload_csv",
                "help": "The PostgreSQL workload to optimize for.",
                "default": None,
            },
            {
                "name": "timeout",
                "long": "timeout",
                "help": "The time allowed for execution before this dodo task will be killed.",
                "default": None,
            },
        ],
    }


def task_project1_enable_logging():
    """
    Project1: enable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='csvlog'",
        "ALTER SYSTEM SET logging_collector='on'",
        "ALTER SYSTEM SET log_statement='all'",
        # For pgreplay.
        "ALTER SYSTEM SET log_connections='on'",
        "ALTER SYSTEM SET log_disconnections='on'",
    ]

    return {
        "actions": [
            *[
                f'PGPASSWORD={DEFAULT_PASS} psql --host=localhost --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_project1_disable_logging():
    """
    Project1: disable logging. (will cause a restart)

    This function will reset to the default parameters on PostgreSQL 14,
    which is not necessarily the right thing to do -- for example, if you
    had custom settings before enable/disable, those custom settings
    will not be restored.
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='stderr'",
        "ALTER SYSTEM SET logging_collector='off'",
        "ALTER SYSTEM SET log_statement='none'",
        # For pgreplay.
        "ALTER SYSTEM SET log_connections='off'",
        "ALTER SYSTEM SET log_disconnections='off'",
    ]

    return {
        "actions": [
            *[
                f'PGPASSWORD={DEFAULT_PASS} psql --host=localhost --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_project1_reset_db():
    """
    Project1: drop (if exists) and create project1db.
    """

    return {
        "actions": [
            # Drop the project database if it exists.
            f"PGPASSWORD={DEFAULT_PASS} dropdb --host=localhost --username={DEFAULT_USER} --if-exists {DEFAULT_DB}",
            # Create the project database.
            f"PGPASSWORD={DEFAULT_PASS} createdb --host=localhost --username={DEFAULT_USER} {DEFAULT_DB}",
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }
