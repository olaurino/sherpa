Feature: Grouping

  Scenario Outline: Simple Grouping
    Given a sherpa session
    And the following x and y arrays
      """
      x = np.arange(1, 101)
      y = np.ones_like(x)
      """
    When I group data with <group counts> counts each
    Then the filtered dependent axis has <final counts> counts in channels
    Examples: Counts
      | group counts  | final counts          |
      | 20            | 20, 20, 20, 20, 20    |
      | 33            | 33, 33, 33, 1         |
