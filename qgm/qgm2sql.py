import json
import argparse
from ops import WHERE_OPS, AGG_OPS, BOX_OPS


def decode_qgm_box(iidx, info, boxes, schema, alias_cnt):
    # Construct from clause
    # get quantifiers with type f
    # get quantifiers with type s and has no predicate
    # get all the join predicates between quantifiers with type f
    # assumption is that if there is a quantifier with type s for the from clause, there are no other quantifiers in the box for the from clause.
    # set dictionary for alias

    select_box = group_box = order_box = None
    for box in boxes:
        if box["operator"] in [
            BOX_OPS.index(key) for key in ["select", "intersect", "except", "union"]
        ]:
            select_box = box
        elif box["operator"] == BOX_OPS.index("groupBy"):
            group_box = box
        elif box["operator"] == BOX_OPS.index("orderBy"):
            order_box = box

    # Construct from clause
    query_from = "FROM "
    alias = {}

    for idx in range(len(select_box["body"]["quantifiers"])):
        quantifier = select_box["body"]["quantifiers"][idx]
        quantifier_type = select_box["body"]["quantifier_types"][idx]

        if quantifier_type == "f":
            table_name = schema["table_names_original"][quantifier]
            if table_name not in alias:
                alias[table_name] = alias_cnt
            else:
                alias[table_name] = (
                    alias[table_name] + [alias_cnt]
                    if isinstance(alias[table_name], list)
                    else [alias[table_name], alias_cnt]
                )
            alias_cnt += 1
            if query_from != "FROM ":
                query_from += " JOIN "
            query_from += table_name + " AS T{}".format(str(alias_cnt - 1))
    if query_from == "FROM ":
        assert len(select_box["body"]["quantifiers"]) == 1
        query_from += (
            "("
            + qgm2sql(
                iidx, info, select_box["body"]["quantifiers"][0], schema, alias_cnt
            )
            + ")"
        )

    local1_cnt = 0
    local2_cnt = 0
    if len(alias) > 1:
        for idx in range(len(select_box["body"]["join_predicates"])):
            col1_id, operator_id, col2_id = select_box["body"]["join_predicates"][idx]
            table1_id, col1_name = schema["column_names_original"][col1_id]
            table2_id, col2_name = schema["column_names_original"][col2_id]
            table1_name = schema["table_names_original"][table1_id]
            table2_name = schema["table_names_original"][table2_id]
            alias1_id = alias[table1_name]
            alias2_id = alias[table2_name]
            if isinstance(alias1_id, list):
                alias1_id = alias1_id[local1_cnt]
                local1_cnt = (local1_cnt + 1) % len(alias[table1_name])
            if isinstance(alias2_id, list):
                alias2_id = alias2_id[local2_cnt]
                local2_cnt = (local2_cnt + 1) % len(alias[table2_name])
            query_from_cond = " ON T{}.{} = T{}.{}".format(
                alias1_id, col1_name, alias2_id, col2_name
            )
            key = max(alias1_id, alias2_id)
            split_idx = query_from.index("T{}".format(key)) + len("T{}".format(key))
            query_from = (
                query_from[:split_idx] + query_from_cond + query_from[split_idx:]
            )

    local_cnt = 0
    query_where = ""
    # Construct where clause
    # get all local predicates and join predicates with quantifier type s
    if select_box["body"]["local_predicates"]:
        for idx in range(len(select_box["body"]["local_predicates"])):
            agg_id, col_id, operator_id, right_operand = select_box["body"][
                "local_predicates"
            ][idx]
            table_id, col_name = schema["column_names_original"][col_id]
            table_name = schema["table_names_original"][table_id]
            alias_id = alias[table_name]
            if isinstance(alias_id, list):
                alias_id = alias_id[local_cnt]
                local_cnt += 1
            operator = WHERE_OPS[operator_id].upper()

            # Right operand
            if isinstance(right_operand, list):
                if isinstance(right_operand[0], dict):
                    right_operand = (
                        "("
                        + qgm2sql(iidx, info, right_operand, schema, alias_cnt)
                        + ")"
                    )
                elif operator_id == WHERE_OPS.index("between"):
                    right_operand = "{} AND {}".format(
                        right_operand[0], right_operand[1]
                    )
                else:
                    agg2_id, col2_id = right_operand
                    table2_id, col2_name = schema["column_names_original"][col2_id]
                    table2_name = schema["table_names_original"][table2_id]
                    alias2_id = alias[table2_name]
                    alias2_id = (
                        alias2_id[local_cnt]
                        if isinstance(alias2_id, list)
                        else alias2_id
                    )
                    right_operand = "T{}.{}".format(alias2_id, col2_name)

            # Create query
            query_where = query_where + " AND " if query_where else "WHERE"
            query_where += " T{}.{} {} {}".format(
                alias_id, col_name, operator, right_operand
            )

    # construct group by clause
    # if box with group by operator
    # head -> group by
    query_group_by = ""
    query_having = ""
    if group_box:
        for head in group_box["head"]:
            agg_id, col_id = head
            agg = AGG_OPS[agg_id]
            table_id, col_name = schema["column_names_original"][col_id]
            table_name = schema["table_names_original"][table_id]

            if table_id == -1:
                agg_col = col_name if agg == "none" else "{}({})".format(agg, col_name)
            else:
                alias_id = alias[table_name]
                agg_col = (
                    "T{}.{}".format(alias_id, col_name)
                    if agg == "none"
                    else "{}(T{}.{})".format(agg, alias_id, col_name)
                )
            query_group_by += (
                ", {}".format(agg_col)
                if query_group_by
                else "GROUP BY {}".format(agg_col)
            )

        # local_predicates -> having
        for local_predicate in group_box["body"]["local_predicates"]:
            agg_id, col_id, operator_id, right_operand = local_predicate
            agg = AGG_OPS[agg_id]
            table_id, col_name = schema["column_names_original"][col_id]
            table_name = schema["table_names_original"][table_id]

            if table_id == -1:
                agg_col = col_name if agg == "none" else "{}({})".format(agg, col_name)
            else:
                alias_id = alias[table_name]
                agg_col = (
                    "T{}.{}".format(alias_id, col_name)
                    if agg == "none"
                    else "{}(T{}.{})".format(agg, alias_id, col_name)
                )

            if operator_id == WHERE_OPS.index("between"):
                right_operand = "{} AND {}".format(right_operand[0], right_operand[1])

            query_having += (
                ", {}".format(agg_col)
                if query_having
                else "HAVING {} {} {}".format(
                    agg_col, WHERE_OPS[operator_id], right_operand
                )
            )

    # Construct order by clause
    # if box with order by operator
    query_order_by = ""
    if order_box:
        assert len(order_box["body"]["quantifier_types"][0]) == 1
        assert order_box["body"]["quantifier_types"][0] == "f"
        for idx in range(len(order_box["head"])):
            agg_id, col_id = order_box["head"][idx]
            agg = AGG_OPS[agg_id]
            table_id, col_name = schema["column_names_original"][col_id]
            table_name = schema["table_names_original"][table_id]

            if table_id == -1:
                agg_col = col_name if agg == "none" else "{}({})".format(agg, col_name)
            else:
                alias_id = alias[table_name]
                agg_col = (
                    "T{}.{}".format(alias_id, col_name)
                    if agg == "none"
                    else "{}(T{}.{})".format(agg, alias_id, col_name)
                )

            direction = "ASC" if order_box["body"]["is_asc"] else "DESC"
            limit_num = order_box["body"]["limit_num"]
            limit_num = " limit {}".format(limit_num) if limit_num else ""
            query_order_by = (
                ", {}".format(agg_col)
                if query_order_by
                else "ORDER BY {}".format(agg_col)
            )
        query_order_by += "{}{}".format(direction, limit_num)

    # Construct select clause
    # read in head of box with operator select
    query_select = ""
    for idx in range(len(select_box["head"])):
        agg_id, col_id = select_box["head"][idx]
        agg = AGG_OPS[agg_id]
        table_id, col_name = schema["column_names_original"][col_id]
        table_name = schema["table_names_original"][table_id]

        if table_id == -1:
            agg_col = col_name if agg == "none" else "{}({})".format(agg, col_name)
        else:
            alias_id = alias[table_name]
            if isinstance(alias_id, list):
                alias_id = alias_id[0]
            agg_col = (
                "T{}.{}".format(alias_id, col_name)
                if agg == "none"
                else "{}(T{}.{})".format(agg, alias_id, col_name)
            )
        query_select += (
            ", {}".format(agg_col) if query_select else "SELECT {}".format(agg_col)
        )

    # Combine
    query = query_select + " " + query_from
    if query_where:
        query += " " + query_where
    if query_group_by:
        query += " " + query_group_by
    if query_having:
        query += " " + query_having
    if query_order_by:
        query += " " + query_order_by

    return query


def qgm2sql(iidx, info, qgm, schema, alias_cnt):
    print(iidx)
    keys = []
    s_idx = []
    for idx, box in enumerate(qgm):
        if box["operator"] in [
            BOX_OPS.index(key) for key in ["intersect", "union", "except"]
        ]:
            s_idx += [idx]
            keys += [BOX_OPS[box["operator"]]]
    if s_idx:
        query = ""
        prev_idx = 0
        for idx, key in enumerate(keys):
            q = decode_qgm_box(
                iidx, info, qgm[prev_idx : s_idx[idx]], schema, alias_cnt
            )
            query += q + " " + key.upper() + " "
            prev_idx = s_idx[idx]
            alias_cnt += 3

        query += decode_qgm_box(iidx, info, qgm[prev_idx:], schema, alias_cnt)
    else:
        query = decode_qgm_box(iidx, info, qgm, schema, alias_cnt)

    return query


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, type=str, help="db tables path")
    parser.add_argument("--source", default=None, type=str, help="loading path")
    parser.add_argument("--destin", default=None, type=str, help="saving path")
    args = parser.parse_args()

    # Load DB info
    dbs = json.load(open(args.db, "rb"))
    db_dic = {db["db_id"]: db for db in dbs}

    # Load data
    datas = json.load(open(args.source, "rb"))

    print("Loading complete!")

    # Convert
    tmp = [
        qgm2sql(idx, data, data["qgm"], db_dic[data["db_id"]], 1)
        for idx, data in enumerate(datas)
    ]

    print("Translating complete!")

    # Save
    with open(args.destin, "w") as f:
        for item in tmp:
            f.write(str(item))
            f.write("\n")

    print("Saving QGM complete!")
