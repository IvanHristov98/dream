@fixture.conn.pool
@db.schema
Feature: Vocabulary tree store

    Background: Vocabulary tree store setup
        Given a vocabulary tree store instance
        And blue tree is created
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
    Scenario: Add new tree while training of another is in progress
        Given blue tree is inserted
        And node Foo is inserted
        And a train job for node Foo is created
        And a train job for node Foo is pushed
        And green tree is created
        Then inserting green tree raises an exception

    @db.cleanup
    Scenario: Insert a third tree
        Given blue tree is inserted
        And green tree is created
        And green tree is inserted
        But pink tree is created
        Then inserting pink tree raises an exception

    @db.cleanup
    Scenario: Fetch missing node
        When node Foo is fetched
        Then a null node is returned

    @db.cleanup
    Scenario Outline: Fetch existing node
        Given blue tree is inserted
        And node <node> is inserted
        When node <node> is fetched
        Then node <node> is returned

        Examples:
            | node |
            | Foo  |
            | Bar  |
            | Baz  |

    @db.cleanup
    Scenario: Update existing node
        Given blue tree is inserted
        And node Bar is inserted
        But Bar is not leaf anymore 
        And Bar is updated
        When node Bar is fetched
        Then node Bar is returned

    @db.cleanup
    Scenario: Update of missing node
        Given Bar is updated
        When node Bar is fetched
        Then a null node is returned

    @db.cleanup
    Scenario: Find existing root
        Given blue tree is inserted
        And node Foo is inserted
        And node Bar is inserted
        When root is fetched
        Then node Foo is returned

    @db.cleanup
    Scenario: Find missing root
        When root is fetched
        Then a null node is returned

    @db.cleanup
    Scenario: Find existing root during training
        Given blue tree is inserted
        And node Foo is inserted
        And a train job for node Foo is created
        And a train job for node Foo is pushed
        When root is fetched
        Then a null node is returned

    @db.cleanup
    Scenario: Find existing root from many trees during no training
        Given blue tree is inserted
        And node Foo is inserted
        And green tree is created
        And green tree nodes are created
            | node | depth |
            | Taz  | 0     |
        And green tree is inserted
        And node Taz is inserted
        When root is fetched
        Then one of nodes is returned
            | node |
            | Foo |
            | Taz |

    @db.cleanup
    Scenario: Find existing root from many trees during training
        Given blue tree is inserted
        And node Foo is inserted
        And green tree is created
        And green tree nodes are created
            | node | depth |
            | Taz  | 0     |
        And green tree is inserted
        And node Taz is inserted
        And a train job for node Taz is created
        And a train job for node Taz is pushed
        When root is fetched
        Then node Foo is returned

    @db.cleanup
    Scenario: Fetch train job during no training
        When train job is fetched
        Then a null train job is returned

    @db.cleanup
    Scenario: Fetch train job during training
        Given blue tree is inserted
        And node Foo is inserted
        And a train job for node Foo is created
        And a train job for node Foo is pushed
        When train job is fetched
        Then a train job for node Foo is returned

    @db.cleanup
    Scenario Outline: Fetch free train job during training while another is being processed
        Given blue tree is inserted
        And node Foo is inserted
        And node Bar is inserted
        And a train job for node Foo is created
        And a train job for node Bar is created
        And a train job for node Foo is pushed
        And a train job for node Bar is pushed
        When <worker_count> train jobs are fetched in parallel txs
        Then train jobs are returned
            | node |
            | Foo  |
            | Bar  |

        Examples:
            | worker_count |
            | 2            |
            | 3            |
            | 4            |

    @db.cleanup
    Scenario: Fetch train job after all have been popped
        Given blue tree is inserted
        And node Foo is inserted
        And a train job for node Foo is created
        And a train job for node Foo is pushed
        But train job for node Foo is popped
        When train job is fetched
        Then a null train job is returned

    @db.cleanup
    Scenario: Fetch train job after some but not all have been popped
        Given blue tree is inserted
        And node Foo is inserted
        And a train job for node Foo is created
        And a train job for node Foo is pushed
        Given node Bar is inserted
        And a train job for node Bar is created
        And a train job for node Bar is pushed
        But train job for node Foo is popped
        When train job is fetched
        Then a train job for node Bar is returned

    @db.cleanup
    Scenario: Cleanup obsolete nodes given no nodes
        Given cleanup is performed
        When root is fetched
        Then a null node is returned

    @db.cleanup
    Scenario: Cleanup obsolete nodes given 1 tree during training
        Given blue tree is inserted
        And node Foo is inserted
        And a train job for node Foo is created
        And a train job for node Foo is pushed
        And cleanup is performed
        When node Foo is fetched
        Then node Foo is returned

    @db.cleanup
    Scenario: Cleanup obsolete nodes given 1 tree during no training
        Given blue tree is inserted
        And node Foo is inserted
        And cleanup is performed
        When node Foo is fetched
        Then node Foo is returned

    @db.cleanup
    Scenario: Cleanup obsolete nodes given 2 trees during training
        Given blue tree is inserted
        And node Foo is inserted
        And green tree is created
        And green tree nodes are created
            | node | depth |
            | Taz  | 0     |
        And green tree is inserted
        And node Taz is inserted
        And a train job for node Taz is created
        And a train job for node Taz is pushed
        And cleanup is performed
        When node Taz is fetched
        Then node Taz is returned

    @db.cleanup
    Scenario: Cleanup obsolete nodes given 2 trees during no training
        Given blue tree is inserted
        And node Foo is inserted
        And green tree is created
        And green tree nodes are created
            | node | depth |
            | Taz  | 0     |
        And green tree is inserted
        And node Taz is inserted
        And cleanup is performed
        When node Foo is fetched
        Then a null node is returned

    @db.cleanup
    Scenario: Cleanup obsolete nodes given 2 trees during no training
        Given blue tree is inserted
        And node Foo is inserted
        And green tree is created
        And green tree nodes are created
            | node | depth |
            | Taz  | 0     |
        And green tree is inserted
        And node Taz is inserted
        And cleanup is performed
        When node Taz is fetched
        Then node Taz is returned
