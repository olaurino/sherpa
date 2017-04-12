Feature: Grouping

  Background:
    Given a sherpa session
    And I import numpy as np
    And np.arange(1, 101) and np.ones_like(x) as x and y arrays

  Scenario Outline: Grouping Counts
    When I group data with <group counts> counts each
    Then the filtered dependent axis has <final counts> counts in channels

    Examples:
      | group counts  | final counts          |
      | 20            | 20, 20, 20, 20, 20    |
      | 33            | 33, 33, 33            |

  Scenario Outline: Grouping Counts and Quality
    When I group data with <group counts> counts each
    Then the dependent axis has a <quality> quality array

    Examples:
      | group counts  | quality                         |
      | 20            | np.zeros_like(x)                |
      | 33            | np.append(np.zeros(x.size-1),2) |

  Scenario Outline: Grouping Counts and Ignore Bad
    When I group data with <group counts> counts each
    And I ignore the bad channels
    Then the filtered dependent axis has <final counts> counts in channels

    Examples:
      | group counts  | final counts          |
      | 20            | 20, 20, 20, 20, 20    |
      | 33            | 33, 33, 33            |

  Scenario: Grouping Counts and Filtering (related to #149)
    When I notice channels from 0 to 50
    And I group data with 20 counts each
    Then the filtered dependent axis has 20, 20 counts in channels

  Scenario: Grouping Counts and Filtering (related to #149)
    When I notice channels from 20 to 70
    And I group data with 20 counts each
    Then the filtered dependent axis has 20, 20 counts in channels

  Scenario: Grouping Counts and Filtering with a response matrix (related to #149)
    Given a simple response matrix
    When I notice channels from 20 to 70
    And I group data with 20 counts each
    Then the filtered dependent axis has 20, 20 counts in channels

  Scenario: Grouping Counts and Filtering with a response matrix in energy space (related to #149)
    Given a simple response matrix
    When I notice energy from 0 to 50
    And I group data with 20 counts each
    Then the filtered dependent axis has 20, 20 counts in channels

  Scenario: Grouping Counts and Filtering with a response matrix in energy space (related to #149)
    Given a simple response matrix
    When I notice energy from 0 to 49
    And I group data with 20 counts each
    Then the filtered dependent axis has 20, 20 counts in channels

  Scenario: Grouping Counts and Filtering with a response matrix in energy space (related to #149)
    Given a simple response matrix
    When I notice energy from 0 to 39
    And I group data with 20 counts each
    Then the filtered dependent axis has 20, 20 counts in channels

  Scenario Outline: Grouping Counts, Filtering, and Ignore Bad (related to #149)
    When I notice channels from 0 to 50
    And I group data with <group counts> counts each
    And I ignore the bad channels
    Then the filtered dependent axis has <final counts> counts in channels

    Examples:
      | group counts  | final counts          |
      | 20            | 20, 20, 20, 20, 20    |
    # The above is true because ignore_bad removes filters
      | 33            | 33, 33, 33            |
    # but bad channels with less counts should be removed as well
