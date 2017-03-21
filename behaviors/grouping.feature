Feature: grouping in sherpa

  Scenario: simple grouping
    Given a sherpa session
    Given a simple array of ones in channels from 1 to 100 loaded in session as PHA
    When I group group data with 20 counts each
    Then the filtered dependent axis has 5 channels with 20 counts each
