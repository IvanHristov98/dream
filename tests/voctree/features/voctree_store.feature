@fixture.conn.pool
@db.schema
Feature: Vocabulary tree store

    Background: Vocabulary tree store setup
        Given a vocabulary tree store instance
        And blue tree nodes are created
            | node | depth |
            | Foo  | 0     |
            | Bar  | 1     |
            | Baz  | 1     |
        And node relations are added
            | child | parent |
            | Bar   | Foo    |
            | Baz   | Foo    |
        And nodes are leafs
            | node |
            | Bar |
            | Baz |

    @db.cleanup
    Scenario: Fetch missing node
        When node Foo is fetched
        Then a null node is returned

    @db.cleanup
    Scenario Outline: Fetch existing node
        Given node <node> is inserted
        When node <node> is fetched
        Then node <node> is returned

        Examples:
            | node |
            | Foo  |
            | Bar  |
            | Baz  |

    @db.cleanup
    Scenario: Update existing node
        Given node Bar is inserted
        But Bar is not leaf anymore 
        And Bar is updated
        When node Bar is fetched
        Then node Bar is returned

    # Scenario: Update of missing node

    # Scenario: Find existing root

    # Scenario: Find missing root
