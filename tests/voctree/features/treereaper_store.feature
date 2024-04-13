@fixture.conn.pool
@db.schema
Feature: Tree reaper

    Background: Frequency store setup
        Given a frequency store instance
        And a vocabulary tree store instance
        And a tree reaper instance
        And term foo is created
        And term bar is created
        And document a is created
        And document b is created
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
    Scenario: Cleanup tree given no nodes
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

    # TODO: Add tests to verify that frequency data is also deleted.