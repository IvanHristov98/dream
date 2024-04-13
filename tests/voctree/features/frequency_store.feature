@fixture.conn.pool
@db.schema
Feature: Frequency store
    
    Background: Frequency store setup
        Given a frequency store instance
        And a vocabulary tree store instance
        And term foo is created
        And document a is created
        And document b is created
        And document c is created
        And green tree is created
        And green tree is inserted

    @db.cleanup
    Scenario: Getting DF of missing term
        When getting df of term foo
        Then no df is returned

    @db.cleanup
    Scenario: Getting DF of existing term
        Given frequencies for term foo are inserted in green tree
            | doc | frequency |
            | a   | 20        |
            | b   | 105       |
            | c   | 1         |
        When getting df of term foo
        Then fetched df has 3 unique docs
        And it has a total tf of 126
    
    @db.cleanup
    Scenario: Getting TFs of existing term
        Given frequencies for term foo are inserted in green tree
            | doc | frequency |
            | a   | 21        |
            | b   | 135       |
        When getting tfs of term foo
        Then it has tfs
            | doc | frequency |
            | a   | 21        |
            | b   | 135       |
        And has no tf for doc c

    @db.cleanup
    Scenario: Getting tree document count when not inserted
        When getting documents count of green tree
        Then it has no docs in tree

    @db.cleanup
    Scenario: Getting tree document count when inserted
        Given blue tree is created
        And blue tree is inserted
        And documents count 15 is inserted for blue tree
        And documents count 1500 is inserted for green tree
        When getting documents count of green tree
        Then it has 1500 docs in tree
        