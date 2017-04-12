Feature: Bug 149 (translation of Cell 3 in Jamie's notebook)

  Scenario: Grouping Counts and Filtering (related to #149)
    Given a sherpa session
    And I load /data/regression_test/master/in/specfit/test_100/acisf00308_000N001_r0044_pha3.fits.gz
    When I notice energy from 0.5 to 7.0
    When I group data with 16 counts each
    Then the filtered dependent axis has 19, 18, 16, 21, 18, 19, 16, 17, 19, 16, 16, 17, 16, 17, 16, 17, 16, 16, 16, 17, 16, 16, 16, 16, 16, 16, 16 counts in channels
