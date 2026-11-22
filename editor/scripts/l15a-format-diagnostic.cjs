const fs = require('node:fs');
const path = require('node:path');
void (async () => {
  const prettier = await import('prettier');
  for (const [index, name] of ['src/main/livePlayControlServer.ts'].entries()) {
    const file = path.resolve(process.cwd(), name);
    const options = (await prettier.resolveConfig(file)) || {};
    const formatted = await prettier.format(fs.readFileSync(file, 'utf8'), { ...options, filepath: file });
    const encoded = Buffer.from(formatted, 'utf8').toString('base64');
    const chunks = encoded.match(/.{1,700}/g) || [];
    console.log('ARC_L15A_TS_START_' + index + '_' + chunks.length);
    for (const [part, chunk] of chunks.entries())
      console.log('ARC_L15A_TS_DATA_' + index + '_' + part + '_' + chunk);
    console.log('ARC_L15A_TS_END_' + index);
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });