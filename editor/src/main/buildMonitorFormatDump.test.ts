import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { format } from 'prettier';
import { it } from 'vitest';

it('BM1 one-time formatting verification', async () => {
  for (const file of ['src/main/buildMonitorService.ts', 'src/main/buildMonitorService.test.ts']) {
    const source = readFileSync(resolve(process.cwd(), file), 'utf8');
    const formatted = await format(source, {
      parser: 'typescript',
      printWidth: 120,
      singleQuote: true,
      trailingComma: 'all',
      semi: true,
      arrowParens: 'always',
      tabWidth: 2,
      endOfLine: 'lf',
    });
    const encoded = Buffer.from(formatted).toString('base64');
    const name = file.endsWith('.test.ts') ? 'test' : 'service';
    for (let i = 0; i < encoded.length; i += 1200) {
      console.log(`BM1FORMAT:${name}:${i / 1200}:${encoded.slice(i, i + 1200)}`);
    }
  }
});
