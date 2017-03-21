Feature: Grouping

  Scenario Outline: grouping
    Given a sherpa session
    And a simple array of ones in channels from 1 to 100 loaded in session as PHA
    When I group data with <group counts> counts each
    Then the filtered dependent axis has <final> channels with <channel counts> each
    Examples: Counts
      | group counts  | final | channel counts |
      | 20            | 5     | 20             |
      | 33            | 3     | 33             |
