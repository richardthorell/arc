const fs = require('node:fs');
const path = require('node:path');

void (async () => {
  const prettier = await import('prettier');
  const names = [
    'src/main/livePlayAuthoringEdits.test.ts',
    'src/main/livePlayAuthoringEdits.ts',
    'src/main/livePlayControlServer.ts',
    'src/main/main.ts',
  ];
  for (const [index, name] of names.entries()) {
    const file = path.resolve(process.cwd(), name);
    const options = (await prettier.resolveConfig(file)) || {};
    const formatted = await prettier.format(fs.readFileSync(file, 'utf8'), { ...options, filepath: file });
    const encoded = Buffer.from(formatted, 'utf8').toString('base64');
    const chunks = encoded.match(/.{1,700}/g) || [];
    process.stdout.write('ARC_FMT_START_' + index + '_' + chunks.length + '\n');
    for (const [part, chunk] of chunks.entries())
      process.stdout.write('ARC_FMT_DATA_' + index + '_' + part + '_' + chunk + '\n');
    process.stdout.write('ARC_FMT_END_' + index + '\n');
  }
})().catch((err) => {
  process.stderr.write(String(err) + '\n');
  process.exitCode = 1;
});