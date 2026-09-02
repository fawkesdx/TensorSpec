from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan


def _row(tag, value, comment=""):
    return (tag, tag, value, comment)


def test_parse_xy_fine_one_loop():
    rows = [
        _row("lwlvnm", "'XY Scan Fine'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "1"),
        _row("scntyp0", "0"),
        _row("devnm_0", "'motors'"),
        _row("nmsbdv0", "2"),
        _row("nm_0_0", "'Scan X'"),
        _row("un_0_0", "'um'"),
        _row("nm_0_1", "'Scan Y'"),
        _row("un_0_1", "'um'"),
        _row("st_0_0", "-40"),
        _row("en_0_0", "40"),
        _row("n_0_0", "81"),
        _row("st_0_1", "-40"),
        _row("en_0_1", "40"),
        _row("n_0_1", "81"),
    ]
    plan = parse_low_level_scan(rows)
    assert plan.mode_name == "XY Scan Fine"
    assert len(plan.loops) == 1
    assert plan.expected_cycles == 81 * 81
    assert plan.has_xy_mesh()


def test_parse_focus_xy_fine_two_loops():
    rows = [
        _row("lwlvnm", "'Focus XY Fine'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "2"),
        _row("nmsbdv0", "1"),
        _row("nm_0_0", "'Slit Defl.'"),
        _row("un_0_0", "'Deg'"),
        _row("st_0_0", "-8.5"),
        _row("en_0_0", "7.5"),
        _row("n_0_0", "17"),
        _row("nmsbdv1", "2"),
        _row("nm_1_0", "'Scan X'"),
        _row("un_1_0", "'um'"),
        _row("nm_1_1", "'Scan Y'"),
        _row("un_1_1", "'um'"),
        _row("st_1_0", "-9.9"),
        _row("en_1_0", "6.0"),
        _row("n_1_0", "81"),
        _row("st_1_1", "-2.7"),
        _row("en_1_1", "13.3"),
        _row("n_1_1", "81"),
    ]
    plan = parse_low_level_scan(rows)
    assert plan.expected_cycles == 17 * 81 * 81
    assert len(plan.angle_motors()) == 1
    assert plan.angle_motors()[0].name == "Slit Defl."
