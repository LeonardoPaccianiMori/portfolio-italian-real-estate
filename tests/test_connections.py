from italian_real_estate.db_urls import get_postgres_connection_string


def test_postgres_url_escapes_credentials():
    url = get_postgres_connection_string(
        host="db.example",
        port="5432",
        user="name@example.com",
        password="p@ss:/word",
        database="warehouse",
    )

    rendered = url.render_as_string(hide_password=False)
    assert "name%40example.com" in rendered
    assert "p%40ss%3A%2Fword" in rendered
    assert rendered.endswith("@db.example:5432/warehouse")
