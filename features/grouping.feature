Feature: Grouping

  Scenario Outline: Grouping Counts
    Given a sherpa session
    And <x> and <y> as x and y arrays
    When I group data with <group counts> counts each
    Then the filtered dependent axis has <final counts> counts in channels

    Examples: Counts
      | x                | y              | group counts  | final counts          |
      | np.arange(1, 101)| np.ones_like(x)| 20            | 20, 20, 20, 20, 20    |
      | np.arange(1, 101)| np.ones_like(x)| 33            | 33, 33, 33, 1         |



  Scenario Outline: Grouping Counts and Quality
    Given a sherpa session
    And <x> and <y> as x and y arrays
    When I group data with <group counts> counts each
    Then the dependent axis has a <quality> quality array

    Examples: Counts
      | x                | y              | group counts  | quality                             |
      | np.arange(1, 101)| np.ones_like(x)| 20            | np.zeros_like(x)                    |
      | np.arange(1, 101)| np.ones_like(x)| 33            | np.append(np.zeros_like(x[0:-1]),2) |
