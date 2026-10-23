#[path = "/Users/drakestapleton/.hermes/cache/scratch/loop21-imap626/crates/spark-mail-rs/src/imap.rs"]
mod imap;
use std::io::{self, Cursor, Read, Write};
struct Stream(Cursor<Vec<u8>>);
impl Read for Stream { fn read(&mut self,b:&mut[u8])->io::Result<usize>{self.0.read(b)} }
impl Write for Stream { fn write(&mut self,b:&[u8])->io::Result<usize>{Ok(b.len())} fn flush(&mut self)->io::Result<()>{Ok(())} }
#[test]
fn uid_inside_quoted_extension_is_not_the_messages_uid() {
    let bytes=b"* OK ready\r\n* 1 FETCH (X-LABEL (\"note UID 9\") UID 10 BODY[HEADER] {3}\r\nbad)\r\nA0001 OK done\r\n".to_vec();
    let mut client=imap::Client::connect(Stream(Cursor::new(bytes))).unwrap();
    assert_eq!(client.uid_fetch(9,imap::Section::Header).unwrap(),None,
        "another message must not be selected by UID text inside its quoted metadata");
}
