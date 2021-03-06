# portfolio_HW
Этот репозиторий предназначен для хранения результатов проекта по курсу "Управление портфелем".
 * Данные: акции, которые хотя бы раз побывали в индексе ММВБ-10. Папка `data`.
 * Задачи: описаны в файле в папке `docs`.
 * Код: `scripts`.
 * Библиотеки: `pyportfolioopt` - требуется предустановка дополнительного софта, смотрите страничку библиотеки, документация там идеальная. Остальные пакеты стандартные.

‼️ **Важный момент: библиотека Pyportfolioopt ставится не очень просто, но она идеально подходит для наших задач. Для её установки необходима Anaconda (с возможностью работать через командную строку), и, конечно, свежий Python>=3.6. Вот как я ставил библиотеку на Windows:**
- ▶️ [По этой ссылке есть инструкции, как ставить библиотеку. Если ОС - Винда, сначала надо установить Visual Studio Installer и настроить его, как и рекомендовано](https://pyportfolioopt.readthedocs.io/en/latest/)
- ▶️ Далее надо поставить на компьютер подходящую версию библиотеки `cvxpy` через Anaconda Prompt: `conda install -c conda-forge cvxpy`.
- ▶️ Наконец, можно поставить сам Pyportfolioopt: `pip install pyportfolioopt` (через `conda` поставить не получится).
