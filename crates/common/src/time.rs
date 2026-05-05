//! Parser de timestamp ISO-8601 minimalista. Aceita exatamente
//! `YYYY-MM-DDTHH:MM:SSZ` (formato fixo do payload da Rinha 2026).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

impl DateTime {
    /// Parse `YYYY-MM-DDTHH:MM:SSZ`. Retorna `None` se fora do formato.
    #[must_use]
    pub fn parse(s: &[u8]) -> Option<Self> {
        if s.len() != 20
            || s[4] != b'-'
            || s[7] != b'-'
            || s[10] != b'T'
            || s[13] != b':'
            || s[16] != b':'
            || s[19] != b'Z'
        {
            return None;
        }
        Some(Self {
            year: parse_n(&s[0..4])? as i32,
            month: parse_n(&s[5..7])? as u8,
            day: parse_n(&s[8..10])? as u8,
            hour: parse_n(&s[11..13])? as u8,
            minute: parse_n(&s[14..16])? as u8,
            second: parse_n(&s[17..19])? as u8,
        })
    }

    /// Dias desde 0000-03-01 (algoritmo days_from_civil de Howard Hinnant —
    /// ver <https://howardhinnant.github.io/date_algorithms.html>).
    #[must_use]
    pub fn days_from_civil(self) -> i64 {
        let y = i64::from(self.year) - i64::from(u8::from(self.month <= 2));
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let m = i64::from(self.month);
        let d = i64::from(self.day);
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// Total de minutos desde epoch (UTC). Aproximação suficiente pro range do
    /// desafio — ignora leap seconds.
    #[must_use]
    pub fn minutes_since_epoch(self) -> i64 {
        let days = self.days_from_civil();
        let secs = days * 86_400
            + i64::from(self.hour) * 3600
            + i64::from(self.minute) * 60
            + i64::from(self.second);
        secs / 60
    }

    /// Dia da semana com segunda=0 ... domingo=6.
    #[must_use]
    pub fn day_of_week_mon0(self) -> u8 {
        // days_from_civil retorna dias desde 1970-01-01 (epoch Unix), que foi
        // uma quinta-feira → mon0=3.
        let d = self.days_from_civil();
        ((d + 3).rem_euclid(7)) as u8
    }
}

fn parse_n(s: &[u8]) -> Option<u32> {
    let mut n: u32 = 0;
    for &b in s {
        if !b.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(u32::from(b - b'0'))?;
    }
    Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_iso_format() {
        let dt = DateTime::parse(b"2026-03-11T20:23:35Z").unwrap();
        assert_eq!(dt.year, 2026);
        assert_eq!(dt.month, 3);
        assert_eq!(dt.day, 11);
        assert_eq!(dt.hour, 20);
        assert_eq!(dt.minute, 23);
        assert_eq!(dt.second, 35);
    }

    #[test]
    fn rejects_malformed() {
        assert!(DateTime::parse(b"2026/03/11T20:23:35Z").is_none());
        assert!(DateTime::parse(b"2026-03-11 20:23:35Z").is_none());
        assert!(DateTime::parse(b"short").is_none());
    }

    #[test]
    fn day_of_week_known_dates() {
        // 2026-03-11 é uma quarta-feira → mon0 = 2.
        let dt = DateTime::parse(b"2026-03-11T00:00:00Z").unwrap();
        assert_eq!(dt.day_of_week_mon0(), 2);
        // 2026-01-01 é uma quinta-feira → mon0 = 3.
        let dt = DateTime::parse(b"2026-01-01T00:00:00Z").unwrap();
        assert_eq!(dt.day_of_week_mon0(), 3);
    }

    #[test]
    fn minutes_diff_known() {
        let a = DateTime::parse(b"2026-03-11T18:45:53Z").unwrap();
        let b = DateTime::parse(b"2026-03-11T14:58:35Z").unwrap();
        let diff = a.minutes_since_epoch() - b.minutes_since_epoch();
        // 18:45 - 14:58 = 3h47 = 227 min (segundos truncados na conversão / 60).
        assert!((226..=228).contains(&diff), "got {diff}");
    }
}
